import random
import numpy as np
from deap import base, creator, tools
from EAController.SleepDataLoader import SleepDataLoader

from ModelController.TrainedModelMaker import TrainedModelMaker
from Globals import Signal, ModelSettings, EvolutionSettings, LoggingSettings, UniquenessFunctions, FitnessFunctions

from EAController.ModifiedEASimple import ModifiedEASimple
from Logs.LogManager import LogManager

class KernelSizeEvolutionaryOptimizer:

    def __init__(self, 
                 sleepstage: str, 
                 signal_type: str):
        
        # Base
        self.sleepstage = sleepstage
        self.signal_type = signal_type

        if ModelSettings.MAX_KERNEL_SIZE == None:
            ModelSettings.MAX_KERNEL_SIZE = self.find_max_kernel_size()
            if ModelSettings.VERBOSE: print(f"Max kernel size set at {ModelSettings.MAX_KERNEL_SIZE}")
        

        self.SDL = SleepDataLoader(
            signal_type=self.signal_type, 
            sleepstage=self.sleepstage,
            batch_size=ModelSettings.BATCH_SIZE)

        if LoggingSettings.LOGGING:
            self.LogManager = LogManager()
        else:
            self.LogManager = None

        self.chosen=[]

        self.setup_deap()
    
    def find_max_kernel_size(self):
        return Signal.SIGNAL_COUNT // 2

    def setup_deap(self):
        """Setup DEAP framework"""
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximizing fitness
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Individual generation
        self.toolbox.register("individual", self.generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", self.select, tournsize=EvolutionSettings.SELECTION_TOURNAMENT_SIZE)

        # Create wrapper functions for evaluation
        def evaluate_normal(individual):
            return self.evaluate_individual(individual, full_training=False)
        
        def evaluate_fully(individual):
            return self.evaluate_individual(individual, full_training=True)
        
        # Genetic operatorss
        self.toolbox.register("evaluate", evaluate_normal)
        self.toolbox.register("fully_evaluate", evaluate_fully)
        
        # Statistics and Hall of Fame
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("med", np.median)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
        self.hall_of_fame = tools.HallOfFame(EvolutionSettings.HALL_OF_FAME_MEMBERS)
    
    def generate_individual(self):
        """Generate a random individual with kernel branches"""

        branches = []

        if ModelSettings.SORT_KERNELS:
            k_max = ModelSettings.MAX_KERNEL_SIZE
            for _ in range(ModelSettings.NUMBER_OF_BRANCHES):
                branch = []
                for _ in range(ModelSettings.KERNELS_PER_BRANCH):
                    kernel = random.randint(ModelSettings.MIN_KERNEL_SIZE, k_max)
                    branch.append(kernel)
                    if kernel < k_max:
                        k_max = kernel

                branches.append(branch)

        else:
            for _ in range( ModelSettings.NUMBER_OF_BRANCHES ):
                branch =  [random.randint(ModelSettings.MIN_KERNEL_SIZE, ModelSettings.MAX_KERNEL_SIZE) for _ in range(ModelSettings.KERNELS_PER_BRANCH)]
                branches.append(branch)

        # Individual format: [[branch1_kernels], [branch2_kernels], ..., [branchN_kernels]]

        individual = []

        for branch in branches:
            individual.append(branch)

        individual = creator.Individual(branches)
        individual.raw_fitness = None
        individual.uniqueness = None
        individual.alpha_beta_fitness = None
        individual.fully_trained = False
        return individual
    
    def evaluate_individual(self, individual, full_training):
        """Evaluate an individual by training a model
        arg: individual"""

        # Train model and get performance
        model_performance = self.create_trained_individual(individual, full_training)
        fitness_value = self.calculate_fitness(model_performance)

        individual.model_performance = model_performance
        individual.raw_fitness = fitness_value
        individual.fully_trained = full_training

        if ModelSettings.VERBOSE:
            print(f"Fitness: {fitness_value}")

        return (fitness_value,)
    
    def create_trained_individual(self, branches: list[list[int]], full_training=False):

        if full_training:
            individual_training_set, individual_test_set, n_samples, pos_weight = self.SDL.get_full_dataset()
            batch_size = EvolutionSettings.TOC_BATCH_SIZE
            epochs = EvolutionSettings.TOC_EPOCHS
            learning_rate = ModelSettings.LEARNING_RATE * EvolutionSettings.TOC_LEARNING_RATE_MULTIPLIER
        else:
            individual_training_set, individual_test_set, n_samples, pos_weight = self.SDL.get_random_subset() 
            batch_size = ModelSettings.BATCH_SIZE
            epochs = ModelSettings.TRAINING_EPOCHS_PER_INDIVIDUAL
            learning_rate = ModelSettings.LEARNING_RATE


        new_model = TrainedModelMaker(
            branches = branches,
            name=f"{branches}, sleepstage: {self.sleepstage}, {batch_size}batch, {ModelSettings.TRAINING_EPOCHS_PER_INDIVIDUAL}epochs",
            sleepstage = self.sleepstage,
            signal_type=self.signal_type,
            batch_size= batch_size,
            train_loader = individual_training_set,
            test_loader = individual_test_set,
            epochs= epochs,
            learning_rate=learning_rate,
            verbose= ModelSettings.VERBOSE,
            N_SAMPLES= n_samples,
            pos_weight= pos_weight,
            have_time_limit = (not full_training)
        )

        return new_model.model_performance

    def calculate_fitness(self, model_performance):
        return FitnessFunctions.fitness_function(model_performance)

    def select(self, individuals, k, tournsize):
        
        if EvolutionSettings.CX_PROB == 0.0 and EvolutionSettings.MUTATION_PROB == 0.0:
            if LoggingSettings.LOGGING:
                for individual in individuals:
                    self.LogManager.check_for_best_in_gen(individual)

            return individuals
        
        self.chosen = []
        for _ in range(k):
            aspirants = [random.choice(individuals) for _ in range(tournsize)]
            next_up = max(aspirants, key=lambda x: self._selection_criteria(x, individuals))
            self.chosen.append(next_up)

        if LoggingSettings.LOGGING:
            for individual in individuals:
                self.LogManager.check_for_best_in_gen(individual)

        return self.chosen

    def _selection_criteria(self, individual, population):

        if FitnessFunctions.normalize[0] == True:
            fitness = FitnessFunctions.normalize[1](individual, population)

        to_be_compared = [ind for ind in self.chosen if ind != individual]
        
        uniqueness = UniquenessFunctions.uniqueness_function(individual, to_be_compared)
        
        print(f"{fitness=}, {uniqueness=}")

        alpha_beta_fitness = (
            EvolutionSettings.alpha * fitness + 
            EvolutionSettings.beta * uniqueness
        )

        individual.uniqueness = uniqueness
        individual.alpha_beta_fitness = alpha_beta_fitness
        individual.fitness.values = (alpha_beta_fitness,)
        
        return alpha_beta_fitness

    def crossover(self, ind1, ind2):
        """Custom crossover for variable-length kernel lists with multiple branches"""

        num_branches = len(ind1)
        assert len(ind2) == num_branches, "Both individuals must have the same number of branches"

        for i in range(num_branches):
            branch1 = ind1[i]
            branch2 = ind2[i]

            if not branch1 or not branch2:
                continue  # Skip empty branches

            for j in range(len(branch1)):
                head1 = branch1[j]
                head2 = branch2[j]

                favorite = random.choice([head1, head2])
                diff = abs(head1 - head2)

                # Gaussian noise
                random_val = min(int(np.floor(abs(np.random.normal(loc=0, scale=4.12)))), 10)
                percentage = random_val / 100.0

                new_head1 = max(ModelSettings.MIN_KERNEL_SIZE, min(int(favorite + percentage * diff), ModelSettings.MAX_KERNEL_SIZE))
                new_head2 = max(ModelSettings.MIN_KERNEL_SIZE, min(int(favorite - percentage * diff), ModelSettings.MAX_KERNEL_SIZE))

                branch1[j] = new_head1
                branch2[j] = new_head2

                if ModelSettings.SORT_KERNELS:
                    branch1.sort(reverse=True)
                    branch2.sort(reverse=True)

            # Update branches
            ind1[i] = branch1
            ind2[i] = branch2

        return ind1, ind2

    def mutate(self, individual):
        """Custom mutation for kernel sizes and branch lengths"""

        mutation_range = 0.2

        for branch in individual:
            for i in range(len(branch)):
                top_of_range = max(2, round(branch[i] * mutation_range))
                delta = random.randint(1, top_of_range)

                if random.random() < 0.5:
                    delta = -delta

                new_value = branch[i] + delta
                new_value = max(ModelSettings.MIN_KERNEL_SIZE, min(ModelSettings.MAX_KERNEL_SIZE, new_value))
                branch[i] = new_value

            if ModelSettings.SORT_KERNELS:
                branch.sort(reverse=True)

        return individual,

    def run_evolution(self):
        """Run the evolutionary algorithm"""
        if ModelSettings.VERBOSE:
            print(f"Starting evolution with {EvolutionSettings.POPULATION_SIZE} individuals for {EvolutionSettings.GENERATIONS} generations")

        # Create initial population
        population = self.toolbox.population(n=EvolutionSettings.POPULATION_SIZE)
        
        # Run evolution
        result_pop = ModifiedEASimple(
            population, 
            self.toolbox,
            cxpb=EvolutionSettings.CX_PROB,
            mutpb=EvolutionSettings.MUTATION_PROB,
            ngen=EvolutionSettings.GENERATIONS,
            LogManager=self.LogManager,
            stats=self.stats,
            halloffame=self.hall_of_fame,
            verbose=ModelSettings.VERBOSE
        )
        
        return result_pop, self.hall_of_fame, self.stats

    def log_results(self):
        
        def get_hall_of_fame_format(i):
            individual = self.hall_of_fame[i]
            return f"{i+1}. Branches={individual}, Fitness={individual.fitness.values[0]:.4f}"

        self.LogManager.log_experiment(
            sleepstage= self.sleepstage,
            signal_type= self.signal_type,
            max_kernel_size= ModelSettings.MAX_KERNEL_SIZE,
            best= get_hall_of_fame_format(0),
            second_best= get_hall_of_fame_format(1),
            third_best= get_hall_of_fame_format(2),
        )
    
    def print_results(self):
        """Print evolution results"""
        print("\n" + "="*50)
        print("EVOLUTION RESULTS")
        print("="*50)
        
        print(f"\nHall of Fame (Top {len(self.hall_of_fame)}):")
        for i, individual in enumerate(self.hall_of_fame):
            print(f"  {i+1}. Branches={individual}, Fitness={individual.fitness.values[0]:.4f}")
            