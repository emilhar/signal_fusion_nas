import random
import torch
import multiprocessing
import numpy as np
from Globals import Globals
from deap import base, creator, tools
from deap.algorithms import eaMuPlusLambda
from dataloaders.data_loader import SDataLoader
from utils.trained_model_maker import TrainedModelMaker
from Globals import EvolutionManager

class KernelSizeEvolutionaryOptimizer:
    MIN_KERNEL_SIZE = 2
    MAX_KERNEL_SIZE = None

    MIN_BRANCHES = 1
    MAX_BRANCHES = 3

    MIN_KERNELS = 2
    MAX_KERNELS = 4

    EPOCHS_PER_INDIVIDUAL = 3

    MINIMIZE_FITNESS = True

    def __init__(self, classification_class, signal_type: str, n_samples:int, batch_size):
        ctx = multiprocessing.get_context('spawn')
        self.evaluation_lock = ctx.Lock()

        self.classification_class = classification_class
        self.signal_type = signal_type
        self.batch_size = batch_size
        self.n_samples = n_samples
        
        self.min_kernel_size, self.max_kernel_size = self.MIN_KERNEL_SIZE, self.MAX_KERNEL_SIZE
        self.min_branches, self.max_branches = self.MIN_BRANCHES, self.MAX_BRANCHES
        self.min_kernels, self.max_kernels = self.MIN_KERNELS, self.MAX_KERNELS

        self.minmax = min if self.MINIMIZE_FITNESS else max
        
        self.epochs_per_individual = self.EPOCHS_PER_INDIVIDUAL

        if self.max_kernel_size is None:
            self.max_kernel_size = n_samples // 2
            print(f"Max kernel size set at {self.max_kernel_size}")
        
        self.SDL = SDataLoader(
            signal_type=self.signal_type, 
            classification_class=self.classification_class,
            batch_size=self.batch_size,
        )

        self.setup_deap()
    
    def setup_deap(self):

        for attr in ['FitnessMinMax', 'Individual']:
            if hasattr(creator, attr):
                delattr(creator, attr)

        if KernelSizeEvolutionaryOptimizer.MINIMIZE_FITNESS:
            creator.create("FitnessMinMax", base.Fitness, weights=(-1.0,))
        else:
            creator.create("FitnessMinMax", base.Fitness, weights=(1.0,))
        
        creator.create("Individual", list, fitness=creator.FitnessMinMax)

        self.toolbox = base.Toolbox()
        
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", self.select)
        self.toolbox.register("evaluate", self.evaluate_individual)
        
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("med", np.median)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
        hall_of_fame_members = 3
        self.hall_of_fame = tools.HallOfFame(hall_of_fame_members)
    
    def get_grid_individuals(self):
        """Only used to 'create' the first generation of individuals"""
        population = [self.generate_individual() for _ in range(EvolutionManager.POPULATION_SIZE)]
        return population
    
    def generate_individual(self):
        CHOICES = [
            max(self.n_samples // 5, 3), 
            max(self.n_samples // 10, 3),
            max(self.n_samples // 30, 3),
            max(self.n_samples // 90, 3),
            max(self.n_samples // 270, 3),
            max(self.n_samples // 810, 3),
            max(self.n_samples // 2430, 3)
        ]
        choices = list(set(x+1 if x%2==0 else x for x in CHOICES))
        choices.sort(reverse=True)
        
        new_indi = []
        
        # Select first number from all available choices
        first_num = random.choice(choices)
        current_max_index = choices.index(first_num)
        
        for _ in range(random.randint(self.min_branches, self.max_branches)):
            branch = []
            for _ in range(random.randint(self.min_kernels, self.max_kernels)):
                # Select from available choices (current_max_index and higher indices)
                available_choices = choices[current_max_index:]
                num = random.choice(available_choices)
                branch.append(num)
                # Update current_max_index - can't go back to earlier indices
                current_max_index = max(current_max_index, choices.index(num))
            
            new_indi.append(branch)
            # Reset current_max_index for the next branch
            current_max_index = choices.index(first_num)
        
        # Individual format: [[branch1_kernels], [branch2_kernels], ..., [branchN_kernels]]
        new_indi = creator.Individual(new_indi)
        
        return new_indi
        
    def evaluate_individual(self, individual, fully=False, _debug=False):
        """Evaluate an individual by training a model
        arg: individual"""

        if _debug:
            individual.model_performance = self._debug_model_performance(individual)

            return (self.fitness_function(individual),)

        model = self.create_trained_individual(individual, fully)
        fitness = self.fitness_function(model.model_performance)

        individual.model_performance = model.model_performance
        individual.model_args = model.model_args

        return (fitness,)
    
    def fitness_function(self, individual_performance):
        # Remember to change MINIMIZE_FITNESS if changing this
        fitness = individual_performance.get("train_loss", 0.0)
        return fitness
    
    def _debug_model_performance(self, indi):
        output = {
            "epoch": 0,
            "train_loss": 0,
            "test_loss": 0,
            "precision": 0,
            "recall": 0,
            "accuracy": 0,
            "lr": 0,
            "branches": indi,
            "best_f1": 0,
            "best_auc": 0,
            "best_true": 0,
            "best_scores": 0,
            "time": 0,
            "state_dict": 0
        }

        return output
    
    def create_trained_individual(self, individual, fully=False):
        """Creates trained individuals. Is used to create all individuals who aren't in the first-generation"""

        if fully:
            epochs = Globals.epochs_for_fully_training_binary_models
            individual_training_set, individual_test_set, n_samples, pos_weight = self.SDL.train_loader, self.SDL.test_loader, self.SDL.n_samples, self.SDL.pos_weight
            batch_size = 128
        else:
            epochs = self.epochs_per_individual
            individual_training_set, individual_test_set, n_samples, pos_weight = self.SDL.get_random_subset(Globals.ea_datapoints_per_individual)
            batch_size = self.batch_size


        new_model = TrainedModelMaker(
            branches=individual,
            N_SAMPLES=n_samples,
            pos_weight=pos_weight,
            train_loader=individual_training_set,
            test_loader=individual_test_set,
            epochs=epochs,
            batch_size=batch_size,    
        )

        return new_model
    
    def select(self, population, number_of_people_to_select):
        """Tournament selection with elitism"""
        
        if number_of_people_to_select <= 0:
            return []
        

        chosen_for_next_generation = []
        elitism = 1

        if elitism > 0:
            sorted_pop = sorted(
                population, 
                key=lambda x: x.fitness.values[0], 
                reverse=not KernelSizeEvolutionaryOptimizer.MINIMIZE_FITNESS
            )
            elites = sorted_pop[:elitism]
            chosen_for_next_generation.extend(elites)
            
            remaining_to_select = number_of_people_to_select - elitism

        else:
            remaining_to_select = number_of_people_to_select
    
        # Perform tournament selection for remaining individuals
        tournsize = max(3, int(len(population) * 0.2))
        for _ in range(remaining_to_select):
            aspirants = [random.choice(population) for _ in range(tournsize)]
            best = self.minmax(aspirants, key= lambda x: x.fitness.values[0])
            chosen_for_next_generation.append(best)
        
        return chosen_for_next_generation

    def crossover(self, ind1, ind2):
        """Custom crossover for variable-length kernel lists with multiple branches"""
        idx1 = random.randint(0, len(ind1) - 1)
        idx2 = random.randint(0, len(ind2) - 1)
        
        ind1[idx1], ind2[idx2] = ind2[idx2], ind1[idx1]

        return ind1, ind2

    def mutate(self, mutant:list[list[int]]):
        """Mutate an individual by randomly modifying branches or kernel sizes."""
        return (mutant,)

        max_number_of_mutations = 3
        mutation_number_options = [i + 1 for i in range(max_number_of_mutations)
              for _ in range(max_number_of_mutations - i)]
        
        number_of_mutations = random.choice(mutation_number_options)

        for _ in range(number_of_mutations):
            mutation_type = self._mutation_choice()
            kernel_size_range = (self.min_kernel_size, self.max_kernel_size)
        
            if mutation_type == "remove_branch":
                if len(mutant) > self.min_branches:
                    mutant.pop(random.randrange(len(mutant)))
            
            elif mutation_type == "add_branch":
                if len(mutant) < self.max_branches:
                    
                    if random.random() < 0.5:
                        mutant.append(
                            [
                                random.randint(*kernel_size_range) 
                                for _ in range(random.randint(self.min_kernels, self.max_kernels))
                            ]
                        )
                        #mutant.append(self.KRNL.theta())
                    else:
                        mutant.append(
                            [
                                random.randint(*kernel_size_range)
                                for _ in range(random.randint(self.min_kernels, self.max_kernels))
                            ]
                        )

            elif mutation_type == "change_branch":
                # Remove
                if len(mutant) > self.min_branches:
                    mutant.pop(random.randrange(len(mutant)))
            
                # Then add (or add an extra branch for single branched individuals)
                if len(mutant) < self.max_branches:
                    if random.random() < 0.5:
                        mutant.append(
                            [
                                random.randint(*kernel_size_range)
                                for _ in range(random.randint(self.min_kernels, self.max_kernels))
                            ]
                        )
                        #mutant.append(self.KRNL.theta())
                    else:
                        mutant.append(
                            [
                                random.randint(*kernel_size_range)
                                for _ in range(random.randint(self.min_kernels, self.max_kernels))
                            ]
                        )

            elif mutation_type == "change_kernel":

                branch_idx = random.randrange(len(mutant))
                kernel_idx = random.randrange(len(mutant[branch_idx]))
                current_value = mutant[branch_idx][kernel_idx]
                
                percentage_change = random.uniform(0.10, 0.20)
                
                if random.random() < 0.5:
                    percentage_change = -percentage_change
                
                # Calculate new value (current_value ± (current_value * percentage_change))
                new_value = current_value + (current_value * percentage_change)
                
                new_value = round(new_value)
                new_value = min(max(new_value, self.min_kernel_size), self.max_kernel_size)
                
                # Apply mutation
                mutant[branch_idx][kernel_idx] = new_value

        return (mutant,)

    def _mutation_choice(self):
        """
        Randomly selects a mutation type based on predefined probability ranges.
        """
        num = random.randint(1, 100)

        if 1 <= num <= 15:       # 15%: Remove branch
            return "remove_branch"
        elif 15 <= num <= 30:    # 15%: Add branch
            return "add_branch"
        elif 30 <= num <= 60:    # 30% Change branch 
            return "change_branch"
        else:                    # 40%: Change kernel
            return "change_kernel"

    def run_evolution(self, part_of_bigger_run=False):
        """Run the evolutionary algorithm"""

        print(f"Starting evolution with {EvolutionManager.POPULATION_SIZE} individuals for {EvolutionManager.GENERATIONS} generations")

        # Create initial population
        population = self.get_grid_individuals()
        

        pop, logbook = eaMuPlusLambda(
            population=population,
            toolbox=self.toolbox,
            mu=EvolutionManager.POPULATION_SIZE,
            lambda_= max(1, EvolutionManager.POPULATION_SIZE // 2),
            cxpb= EvolutionManager.CX_PROB,
            mutpb= EvolutionManager.MUTATION_PROB,
            ngen= EvolutionManager.GENERATIONS,
            stats=self.stats,
            halloffame= self.hall_of_fame,
        )

        if part_of_bigger_run:
            best_individual = self.hall_of_fame[0]
            a = best_individual.model_performance
            with self.evaluation_lock:
                print(f"Fully training best individual for {self.classification_class} with {self.signal_type}")
                self.evaluate_individual(best_individual, fully=True)
            b = best_individual.model_performance

            if a == b:
                raise ValueError("yo mama")

            temp_dir = "temp_models"

            torch.save(
                {
                    "state_dict": best_individual.model_performance["state_dict"],
                    "model_args": best_individual.model_args,
                },
                f"{temp_dir}/{self.classification_class.given_name}_{self.signal_type}_model.pt"
            )

        print(f"Best individual for {self.classification_class} with {self.signal_type}: {self.hall_of_fame[0]} with fitness = {self.hall_of_fame[0].fitness.values[0]}")

        return logbook