import random
import torch
import numpy as np
from deap import base, creator, tools
from EAController.SleepDataLoader import SleepDataLoader

from ModelController.TrainedModelMaker import TrainedModelMaker
from Globals import Signal, ModelManager, EvolutionManager, AlpsManager, LoggingSettings, LoggingTemplate, FitnessFunctions, TimeWall

from EAController.SLeaMuPlusLambda import SLeaMuPlusLambda
from Logs.LogManager import LogManager

class KernelSizeEvolutionaryOptimizer:

    def __init__(self, sleepstage: str, signal_type: str):
        
        # Base
        self.sleepstage = sleepstage
        self.signal_type = signal_type

        if ModelManager.MAX_KERNEL_SIZE == None:
            ModelManager.MAX_KERNEL_SIZE = self.find_max_kernel_size()
            if EvolutionManager.VERBOSE: print(f"Max kernel size set at {ModelManager.MAX_KERNEL_SIZE}")
        
        self.SDL = SleepDataLoader(
            signal_type=self.signal_type, 
            sleepstage=self.sleepstage)

        if LoggingSettings.LOGGING:
            self.LogManager = LogManager()
        else:
            self.LogManager = None

        self.chosen_for_next_generation = []

        self.setup_deap()
    
    def find_max_kernel_size(self):
        return Signal.SIGNAL_COUNT // 2

    def setup_deap(self):
        """Setup DEAP framework"""
        # Create fitness and individual classes
        if FitnessFunctions.MINIMIZE_FITNESS:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
        else:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        
        # Individual generation
        self.toolbox.register("individual", self.generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", self.select, tournsize=EvolutionManager.SELECTION_TOURNAMENT_SIZE)
        
        # Genetic operatorss
        self.toolbox.register("evaluate", self.evaluate_individual)
        
        # Statistics and Hall of Fame
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("med", np.median)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
        self.hall_of_fame = tools.HallOfFame(EvolutionManager.HALL_OF_FAME_MEMBERS)
    
    def generate_individual(self):
        """Generate a random individual with kernel branches.
        Only used to create the first generation of individuals"""

        branches = []
        number_of_branches = random.randint( ModelManager.NUMBER_OF_BRANCHES_RANGE[0], ModelManager.NUMBER_OF_BRANCHES_RANGE[1])
        kernel_per_branch = [ random.randint( ModelManager.NUMBER_OF_KERNELS_RANGE[0], ModelManager.NUMBER_OF_KERNELS_RANGE[1]) for _ in range(number_of_branches) ]

        branches = [[random.randint(ModelManager.MIN_KERNEL_SIZE, ModelManager.MAX_KERNEL_SIZE) for _ in range(kernel_per_branch[i])] for i in range(number_of_branches)]

        # Individual format: [[branch1_kernels], [branch2_kernels], ..., [branchN_kernels]]

        individual = creator.Individual(branches)
        individual.age =  0
        individual.layer = 0

        return individual
    
    def evaluate_individual(self, individual):
        """Evaluate an individual by training a model
        arg: individual"""

        model_performance = self.create_trained_individual(individual, individual.layer)
        fitness = self.calculate_fitness(model_performance)

        individual.model_performance = model_performance
        individual.individual_id = LoggingSettings.current_individual_id
        LoggingSettings.current_individual_id += 1

        return (fitness,)
    
    def create_trained_individual(self, individual, layer):
        """Creates trained individuals. Is used to create all individuals who aren't in the first-generation"""

        individual_training_set, individual_test_set, n_samples, pos_weight = self.SDL.get_random_subset(
            dataset_percentage = AlpsManager.TRAINING_SETTINGS_FOR_LAYERS[layer]["dataset_percentage"],
            batch_size=AlpsManager.TRAINING_SETTINGS_FOR_LAYERS[layer]["batch_size"]) 
        
        batch_size = AlpsManager.TRAINING_SETTINGS_FOR_LAYERS[layer]["batch_size"]
        epochs =AlpsManager.TRAINING_SETTINGS_FOR_LAYERS[layer]["training_epochs"]
        learning_rate = ModelManager.LEARNING_RATE

        

        new_model = TrainedModelMaker(
            branches = individual,
            name=f"{individual}, {batch_size}batch, {epochs}epochs",
            sleepstage = self.sleepstage,
            signal_type=self.signal_type,
            batch_size= batch_size,
            train_loader = individual_training_set,
            test_loader = individual_test_set,
            epochs= epochs,
            learning_rate=learning_rate,
            N_SAMPLES= n_samples,
            pos_weight= pos_weight,
        )

        return new_model.model_performance

    def calculate_fitness(self, model_performance):
        return FitnessFunctions.fitness_function(model_performance)

    def select(self, population, number_of_people_to_select, tournsize):
        """Tournament selection with elitism"""
        
        if number_of_people_to_select <= 0:
            return []
        
        # Normalize fitness values
        map(lambda x: FitnessFunctions.normalization_function(x, population), population)

        self.chosen_for_next_generation = []
        elitism = EvolutionManager.ELITISM

        # Elitism: preserve the best individuals
        if elitism > 0:
            # Sort population based on fitness (ascending for minimization, descending for maximization)
            sorted_pop = sorted(population, key=lambda x: x.fitness.values[0], reverse=not FitnessFunctions.MINIMIZE_FITNESS)
            elites = sorted_pop[:elitism]
            self.chosen_for_next_generation.extend(elites)
            
            # Adjust number of individuals to select through tournament
            remaining_to_select = number_of_people_to_select - elitism

        else:
            remaining_to_select = number_of_people_to_select
        
        # Time wall
        if TimeWall.ON and population[0].layer != 0:
            print(f"""\n\nTIME WALL RANK FOR LAYER {population[0].layer}""")
            population = sorted(population, reverse=True, key=lambda x: x.model_performance[LoggingTemplate.time])
            for i, guy in enumerate(population):
                print(i+1,guy, guy.model_performance[LoggingTemplate.time]) 

            n = int(len(population) * TimeWall.time_wall_percentage)
            population = population[n:]

            print("\nTIME WALL AFTER")
            for i, guy in enumerate(population):
                print(i+1,guy, guy.model_performance[LoggingTemplate.time]) 

        # Perform tournament selection for remaining individuals
        for _ in range(remaining_to_select):
            aspirants = [random.choice(population) for _ in range(tournsize)]
            best = max(aspirants, key= lambda x: x.fitness.values[0])
            self.chosen_for_next_generation.append(best)
        
        return self.chosen_for_next_generation

    def crossover(self, ind1, ind2):
        """Custom crossover for variable-length kernel lists with multiple branches"""
        idx1 = random.randint(0, len(ind1) - 1)
        idx2 = random.randint(0, len(ind2) - 1)
        
        ind1[idx1], ind2[idx2] = ind2[idx2], ind1[idx1]

        return ind1, ind2

    def mutate(self, individual):
        """Mutate an individual by randomly modifying branches or kernel sizes."""
        
        # Clone the individual to avoid in-place issues
        mutant: list[list] = creator.Individual([branch[:] for branch in individual])
        mutant.age = individual.age
        mutant.layer = individual.layer

        mutation_number_options = [i + 1 for i in range(EvolutionManager.MAX_NUMBER_OF_MUTATIONS)
              for _ in range(EvolutionManager.MAX_NUMBER_OF_MUTATIONS - i)]
        
        number_of_mutations = random.choice(mutation_number_options)

        for _ in range(number_of_mutations):
            mutation_type = self._mutation_choice()
        
            if mutation_type == "remove_branch":
                if len(mutant) > ModelManager.NUMBER_OF_BRANCHES_RANGE[0]:
                    mutant.pop(random.randrange(len(mutant)))
            
            elif mutation_type == "add_branch":
                if len(mutant) < ModelManager.NUMBER_OF_BRANCHES_RANGE[1]:
                    branch_length = random.randint(*ModelManager.NUMBER_OF_KERNELS_RANGE)
                    first_kernel = max(1, random.choice(range(ModelManager.MIN_KERNEL_SIZE, ModelManager.MAX_KERNEL_SIZE, 20))-1)
                    new_branch = [first_kernel]
                    for _ in range(branch_length - 1):
                        new_branch.append(max( new_branch[-1] // 2, 1))
                    mutant.append(new_branch)

            elif mutation_type == "add_kernel":
                branch_idx = random.randrange(len(mutant))
                if len(mutant[branch_idx]) < ModelManager.NUMBER_OF_KERNELS_RANGE[1]:
                    new_kernel = random.randint(ModelManager.MIN_KERNEL_SIZE, ModelManager.MAX_KERNEL_SIZE)
                    nk_index = random.randint(0, len(mutant[branch_idx])-1)
                    
                    mutant[branch_idx].insert(nk_index, new_kernel)

            elif mutation_type == "change_kernel":

                branch_idx = random.randrange(len(mutant))
                kernel_idx = random.randrange(len(mutant[branch_idx]))
                current_value = mutant[branch_idx][kernel_idx]
                
                # Get random percentage change between 10% and 20% (0.10 to 0.20)
                percentage_change = random.uniform(0.10, 0.20)
                
                # Randomly decide whether to increase or decrease
                if random.random() < 0.5:
                    percentage_change = -percentage_change
                
                # Calculate new value (current_value ± (current_value * percentage_change))
                new_value = current_value + (current_value * percentage_change)
                
                # Round to nearest integer and clamp to valid kernel sizes
                new_value = int(round(new_value))
                new_value = min(max(new_value, ModelManager.MIN_KERNEL_SIZE), ModelManager.MAX_KERNEL_SIZE)
                
                # Apply mutation
                mutant[branch_idx][kernel_idx] = new_value

        return (mutant,)

    def _mutation_choice(self):
        """
        Randomly selects a mutation type based on predefined probability ranges.
        """
        num = random.randint(1, 100)

        if 1 <= num <= 5:       # 5%: Remove branch
            return "remove_branch"
        elif 5 <= num <= 20:    # 15%: Add branch
            return "add_branch"
        elif 20 <= num <= 45:    # 25%: Add kernel
            return "add_kernel"
        else:                    # 55%: Change kernel
            return "change_kernel"

    def run_evolution(self, logging_folder_for_omega_runs=None):
        """Run the evolutionary algorithm"""
        if EvolutionManager.VERBOSE:
            print(f"Starting evolution with {EvolutionManager.POPULATION_SIZE_PER_LAYER} individuals for {EvolutionManager.GENERATIONS} generations")

        # Create initial population
        population = self.toolbox.population(n=EvolutionManager.POPULATION_SIZE_PER_LAYER)

        mu= EvolutionManager.POPULATION_SIZE_PER_LAYER
        lambda_ = mu // 2
        
        # Run evolution
        evolver = SLeaMuPlusLambda(
            population=population,
            toolbox=self.toolbox,
            mu= mu,
            lambda_ = lambda_,
            halloffame= self.hall_of_fame,
            LogManager= self.LogManager,
        )
        
        result_pop = evolver.main(
            stats= self.stats,
        )

        if LoggingSettings.LOGGING:
            self.log_results(logging_folder_for_omega_runs)

        self.print_results()
        
        return result_pop, self.hall_of_fame, self.stats

    def log_results(self, logging_folder_for_omega_runs):
        
        def get_hall_of_fame_format(i):
            individual = self.hall_of_fame[i]
            return f"{i+1}. Branches={individual}, Fitness={individual.fitness.values[0]:.4f}"

        self.LogManager.log_experiment(
            sleepstage= self.sleepstage,
            signal_type= self.signal_type,
            max_kernel_size= ModelManager.MAX_KERNEL_SIZE,
            best= get_hall_of_fame_format(0),
            second_best= get_hall_of_fame_format(1),
            third_best= get_hall_of_fame_format(2),
        )

        if logging_folder_for_omega_runs:
            best_individual = self.hall_of_fame[0]
            torch.save(best_individual.model_performance[LoggingTemplate.state_dict], logging_folder_for_omega_runs + "/" + f"{self.sleepstage}-{self.signal_type}")
        
    def print_results(self):
        """Print evolution results"""
        print("\n" + "="*50)
        print("EVOLUTION RESULTS")
        print("="*50)
        
        print(f"\nHall of Fame (Top {len(self.hall_of_fame)}):")
        for i, individual in enumerate(self.hall_of_fame):
            print(f"  {i+1}. Branches={individual}, Fitness={individual.fitness.values[0]:.4f}")
