import os
import random
import torch
import numpy as np
from deap import base, creator, tools
from data.data_loader import SDataLoader
from ea_controller.ea_algorithm import EA_Algorithm

from ea_controller.trained_model_maker import TrainedModelMaker
from Globals import Signal, EvolutionManager, LoggingSettings, LoggingTemplate, FitnessFunctions

from logs.log_manager import LogManager

class KernelSizeEvolutionaryOptimizer:
    MIN_KERNEL_SIZE = 2
    MAX_KERNEL_SIZE = None

    MIN_BRANCHES = 1
    MAX_BRANCHES = 3

    MIN_KERNELS = 2
    MAX_KERNELS = 4

    EPOCHS_PER_INDIVIDUAL = 1
    LEARNING_RATE = 5e-4

    def __init__(self, classification_class: str, signal_type: str, batch_size, n_samples):

        self.classification_class = classification_class
        self.signal_type = signal_type
        self.batch_size = batch_size
        self.n_samples = n_samples
        
        self.min_kernel_size, self.max_kernel_size = self.MIN_KERNEL_SIZE, self.MAX_KERNEL_SIZE
        self.min_branches, self.max_branches = self.MIN_BRANCHES, self.MAX_BRANCHES
        self.min_kernels, self.max_kernels = self.MIN_KERNELS, self.MAX_KERNELS
        
        self.epochs_per_individual = self.EPOCHS_PER_INDIVIDUAL
        self.learning_rate = self.LEARNING_RATE

        if self.max_kernel_size is None:
            self.max_kernel_size = Signal.SIGNAL_COUNT // 2
            if EvolutionManager.VERBOSE: print(f"Max kernel size set at {self.max_kernel_size}")
        
        self.SDL = SDataLoader(
            signal_type=self.signal_type, 
            classification_class=self.classification_class,
            batch_size=self.batch_size,
        )

        if LoggingSettings.LOGGING:
            self.log_manager = LogManager()
        else:
            self.log_manager = None

        #self.KRNL = KRNL_GridSearch(self.signal_type, self.classification_class)

        self.setup_deap()
    
    def setup_deap(self):
        """Setup DEAP framework"""

        for attr in ['FitnessMinMax', 'Individual']:
            if hasattr(creator, attr):
                delattr(creator, attr)

        # Create fitness and individual classes
        if FitnessFunctions.MINIMIZE_FITNESS:
            creator.create("FitnessMinMax", base.Fitness, weights=(-1.0,))
        else:
            creator.create("FitnessMinMax", base.Fitness, weights=(1.0,))
        
        creator.create("Individual", list, fitness=creator.FitnessMinMax)

        self.toolbox = base.Toolbox()
        
        # Genetic operators
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", self.select, tournsize=EvolutionManager.SELECTION_TOURNAMENT_SIZE)
        self.toolbox.register("evaluate", self.evaluate_individual)
        
        # Statistics and Hall of Fame
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("med", np.median)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
        self.hall_of_fame = tools.HallOfFame(EvolutionManager.HALL_OF_FAME_MEMBERS)
    
    def get_grid_individuals(self):
        """Only used to 'create' the first generation of individuals"""
        population = [self.generate_individual() for _ in range(EvolutionManager.POPULATION_SIZE)]
        return population
    
    def generate_individual(self):
        new_indi = []

        for _ in range(random.randint(self.min_branches, self.max_branches)):
            new_indi.append(
                [
                    random.randint(self.min_kernel_size, self.max_kernel_size)
                    for _ in range(random.randint(self.min_kernels, self.max_kernels))
                ]
            )
            #new_indi.append(self.KRNL.theta())

        # Individual format: [[branch1_kernels], [branch2_kernels], ..., [branchN_kernels]]
        new_indi = creator.Individual(new_indi)
        
        return new_indi
        
    def evaluate_individual(self, individual, _debug=False):
        """Evaluate an individual by training a model
        arg: individual"""

        if _debug:
            individual.model_performance = self._debug_model_performance(individual)
            individual.individual_id = LoggingSettings.current_individual_id
            LoggingSettings.current_individual_id += 1

            return (FitnessFunctions.fitness_function(individual),)

        model = self.create_trained_individual(individual)
        fitness = FitnessFunctions.fitness_function(model.model_performance)

        individual.model_performance = model.model_performance
        individual.model_args = model.model_args

        individual.individual_id = LoggingSettings.current_individual_id
        LoggingSettings.current_individual_id += 1

        return (fitness,)
    
    def _debug_model_performance(self, indi):
        it = LoggingTemplate
        output = {
            it.epoch: 0,
            it.train_loss: 0,
            it.test_loss: 0,
            it.precision: 0,
            it.recall: 0,
            it.accuracy: 0,
            it.lr: 0,
            it.branches: indi,
            it.best_f1: 0,
            it.best_auc: 0,
            it.best_true: 0,
            it.best_scores: 0,
            it.time: 0,
            it.state_dict: 0
        }
        return output
    
    def create_trained_individual(self, individual):
        """Creates trained individuals. Is used to create all individuals who aren't in the first-generation"""

        individual_training_set, individual_test_set, n_samples, pos_weight = self.SDL.get_random_subset() 

        new_model = TrainedModelMaker(
            branches=individual,
            N_SAMPLES=n_samples,
            pos_weight=pos_weight,
            train_loader=individual_training_set,
            test_loader=individual_test_set,
            batch_size=self.batch_size,
            epochs=self.epochs_per_individual,
            learning_rate=self.learning_rate,
        )

        return new_model
    
    def select(self, population, number_of_people_to_select, tournsize):
        """Tournament selection with elitism"""
        
        if number_of_people_to_select <= 0:
            return []
        
        # Normalize fitness values
        map(lambda x: FitnessFunctions.normalization_function(x, population), population)

        chosen_for_next_generation = []
        elitism = EvolutionManager.ELITISM

        if elitism > 0:
            sorted_pop = sorted(
                population, 
                key=lambda x: x.fitness.values[0], 
                reverse=not FitnessFunctions.MINIMIZE_FITNESS
            )
            elites = sorted_pop[:elitism]
            chosen_for_next_generation.extend(elites)
            
            remaining_to_select = number_of_people_to_select - elitism

        else:
            remaining_to_select = number_of_people_to_select
    
        # Perform tournament selection for remaining individuals
        for _ in range(remaining_to_select):
            aspirants = [random.choice(population) for _ in range(tournsize)]
            best = max(aspirants, key= lambda x: x.fitness.values[0])
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

        mutation_number_options = [i + 1 for i in range(EvolutionManager.MAX_NUMBER_OF_MUTATIONS)
              for _ in range(EvolutionManager.MAX_NUMBER_OF_MUTATIONS - i)]
        
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

        if EvolutionManager.VERBOSE:
            print(
                f"Starting evolution with {EvolutionManager.POPULATION_SIZE}" 
                "individuals for {EvolutionManager.GENERATIONS} generations"
            )

        # Create initial population
        population = self.get_grid_individuals()
        
        # Run evolution
        algorithm = EA_Algorithm(
            population=population,
            toolbox=self.toolbox,
            halloffame= self.hall_of_fame,
            log_manager= self.log_manager,
        )
        
        algorithm.eaMuPlusLambda(
            stats= self.stats,
        )

        self.log_results(part_of_bigger_run)
        self.print_results()

    def log_results(self, part_of_bigger_run):
        
        def get_hall_of_fame_format(i):
            individual = self.hall_of_fame[i]
            return f"{i+1}. Branches={individual}, Fitness={individual.fitness.values[0]:.4f}"

        self.log_manager.log_experiment(
            classification_class= self.classification_class,
            signal_type= self.signal_type,
            max_kernel_size= self.max_kernel_size,
            best= get_hall_of_fame_format(0),
            second_best= get_hall_of_fame_format(1),
            third_best= get_hall_of_fame_format(2),
        )

        if part_of_bigger_run:
            best_individual = self.hall_of_fame[0]
            torch.save(
            {
                "state_dict": best_individual.model_performance[LoggingTemplate.state_dict],
                "model_args": best_individual.model_args,
            },
            os.path.join(f"ea_controller/saved_models/Experiment_{self.log_manager.experiment_id}", f"{self.classification_class}_{self.signal_type}_classifier.pt"))

    def print_results(self):
        """Print evolution results in a dynamically sized table"""
        title = "EVOLUTION RESULTS"
        border = "=" * 80
        
        max_branch_len = max(len(str(ind)) for ind in self.hall_of_fame) if self.hall_of_fame else 20
        max_branch_len = min(max_branch_len, 50)
        
        rank_width = 6
        fitness_width = 12
        branches_width = max_branch_len + 2
        
        # Header
        print("\n\n" + border)
        print(title.center(len(border)))
        print(border)
        
        # Table
        print(f"\nHall of Fame (Top {len(self.hall_of_fame)}):")
        print("-" * (rank_width + branches_width + fitness_width + 6))  # +6 for separators
        
        # Column headers (dynamically spaced)
        header = (
            f"{'Rank':<{rank_width}} | "
            f"{'Branches':<{branches_width}} | "
            f"{'Fitness':<{fitness_width}}"
        )
        print(header)
        print("-" * (rank_width + branches_width + fitness_width + 6))
        
        # Rows
        for i, ind in enumerate(self.hall_of_fame):
            # Truncate long branch strings if needed
            branches_str = str(ind)
            if len(branches_str) > branches_width:
                branches_str = branches_str[:branches_width-3] + "..."
            
            row = (
                f"{i+1:<{rank_width}} | "
                f"{branches_str:<{branches_width}} | "
                f"{ind.fitness.values[0]:<{fitness_width}.4f}"
            )
            print(row)
        
        print("-" * (rank_width + branches_width + fitness_width + 6))