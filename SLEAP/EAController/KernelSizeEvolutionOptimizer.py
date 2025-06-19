import random
import numpy as np
from deap import base, creator, tools
from EAController.SleepDataLoader import SleepDataLoader

from ModelController.TrainedModelMaker import TrainedModelMaker
from Globals import Signal, ModelSettings, EvolutionSettings, LoggingSettings

from EAController.ModifiedEASimple import ModifiedEASimple
from Logs.LogManager import LogManager

class KernelSizeEvolutionaryOptimizer:
    def __init__(self, 
                 
                 # Base
                 sleepstage: str, 
                 signal_type: str,
                 batch_size: int = ModelSettings.BATCH_SIZE,
                 epochs_per_individual: int = ModelSettings.TRAINING_EPOCHS_PER_INDIVIDUAL,
                 dataset_fraction: float = ModelSettings.DATASET_FRACTION,
                 
                 # Evolution parameters
                 population_size: int = EvolutionSettings.POPULATION_SIZE,
                 generations: int = EvolutionSettings.GENERATIONS,
                 cx_prob: float = EvolutionSettings.CX_PROB,
                 mut_prob: float = EvolutionSettings.MUTATION_PROB,
                 tournament_size: int = EvolutionSettings.TOURNAMENT_SIZE,
                 
                 # Kernel size constraints
                 min_kernel_size: int = ModelSettings.MIN_KERNEL_SIZE,
                 max_kernel_size: int|None = ModelSettings.MAX_KERNEL_SIZE,
                 
                 verbose: bool = ModelSettings.VERBOSE):
        
        # Base
        self.sleepstage = sleepstage
        self.signal_type = signal_type
        self.batch_size = batch_size
        self.dataset_fraction = dataset_fraction
        self.epochs = epochs_per_individual
        
        # Evolution parameters
        self.population_size = population_size
        self.generations = generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.tournament_size = tournament_size
        
        # Kernel constraints
        self.min_kernel_size = min_kernel_size
        
        if max_kernel_size == None:
            self.max_kernel_size = self.find_max_kernel_size()
            if verbose: print(f"Max kernel size set at {self.max_kernel_size}")
        else:
            self.max_kernel_size = max_kernel_size
        
        self.verbose = verbose

        self.SDL = SleepDataLoader(verbose=self.verbose, 
        signal_type=self.signal_type, 
        sleepstage=self.sleepstage,
        dataset_fraction=self.dataset_fraction,
        batch_size=self.batch_size)

        self.LogManager = LogManager()
        

        self.setup_deap()
    
    def find_max_kernel_size(self):
        if self.signal_type == Signal.EMG.SUBMENTAL:
            return (Signal.EMG_SIGNAL_COUNT // 2)
        else:
            return (Signal.NON_EMG_SIGNAL_COUNT // 2)

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
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        # Create wrapper functions for evaluation
        def evaluate_normal(individual):
            return self.evaluate_individual(individual, champion=False)
        
        def evaluate_champion(individual):
            return self.evaluate_individual(individual, champion=True)
        
        # Genetic operators
        self.toolbox.register("evaluate", evaluate_normal)
        self.toolbox.register("evaluate_champion", evaluate_champion)
        
        # Statistics and Hall of Fame
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("med", np.median)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
        self.hall_of_fame = tools.HallOfFame(EvolutionSettings.HALL_OF_FAME_MEMBERS)
    
    def generate_individual(self):
        """Generate a random individual with left and right kernel branches"""

        left_kernels = [random.randint(self.min_kernel_size, self.max_kernel_size) for _ in range(EvolutionSettings.RANDOM_KERNELS_PER_BRANCH)]
        right_kernels = [random.randint(self.min_kernel_size, self.max_kernel_size) for _ in range(EvolutionSettings.RANDOM_KERNELS_PER_BRANCH)]

        left_kernels += ModelSettings.KERNEL[ (EvolutionSettings.LAYERS_OF_CNN - EvolutionSettings.RANDOM_KERNELS_PER_BRANCH) - 1 :]
        right_kernels += ModelSettings.KERNEL[ (EvolutionSettings.LAYERS_OF_CNN - EvolutionSettings.RANDOM_KERNELS_PER_BRANCH) - 1 :]
        
        # Individual format: [left_branch_length, *left_kernels, right_branch_length, *right_kernels]
        individual = [EvolutionSettings.LAYERS_OF_CNN] + left_kernels + [EvolutionSettings.LAYERS_OF_CNN] + right_kernels
        
        return creator.Individual(individual)
    
    def decode_individual(self, individual):
        """Decode individual into left and right kernel lists"""
        left_length = individual[0]
        left_kernels = individual[1:1+left_length]
        
        right_start = 1 + left_length
        right_length = individual[right_start]
        right_kernels = individual[right_start+1:right_start+1+right_length]
        
        return left_kernels, right_kernels
    
    def evaluate_individual(self, individual, champion=False):
        """Evaluate an individual by training a model
        arg: individual
        arg: champion Bool, if individual is partaking in a tournament of champions, then they train on the full dataset"""

        left_kernels, right_kernels = self.decode_individual(individual)
        
        # Train model and get performance
        model_performance = self.create_individual(left_kernels, right_kernels, champion)
        
        fitness_value = self.calculate_fitness(model_performance)
        
        if self.verbose:
            print(f"Fitness: {fitness_value}")

        if EvolutionSettings.LOGGING:

            train_loss = model_performance.get("Train Loss", 0.0),
            test_loss = model_performance.get("Test Loss", 0.0),
            precision = model_performance.get("Precision", 0.0),
            recall = model_performance.get("Recall", 0.0),
            f1 = model_performance.get("F1", 0.0),
            accuracy = model_performance.get("Accuracy", 0.0),

            self.LogManager.check_for_best_in_gen(individual, fitness_value, champion, train_loss, test_loss, precision, recall, f1, accuracy)
        
        return (fitness_value,)
    
    def create_individual(self, left_kernel_sizes:list[int], right_kernel_sizes: list[int], champion=False):

        if champion:
            individual_training_set, individual_test_set, n_samples, pos_weight = self.SDL.get_full_dataset()
        else:
            individual_training_set, individual_test_set, n_samples, pos_weight = self.SDL.get_random_subset() 

        # Things marked with # come from the SDL
        new_model = TrainedModelMaker(
            left_kernel_sizes=left_kernel_sizes, #
            right_kernel_sizes=right_kernel_sizes, #
            name=f"{left_kernel_sizes} :-: {right_kernel_sizes}, sleepstage: {self.sleepstage}, {self.batch_size}batch, {self.epochs}epochs",
            sleepstage = self.sleepstage,
            signal_type=self.signal_type,
            batch_size= self.batch_size,
            train_loader = individual_training_set,
            test_loader = individual_test_set,
            epochs= self.epochs,
            verbose= self.verbose,
            N_SAMPLES= n_samples, #
            pos_weight= pos_weight) #

        return new_model.model_performance

    def calculate_fitness(self, model_performance):
        f1_score = model_performance.get("F1", 0.0)
        return f1_score
    
    def crossover(self, ind1, ind2):
        """Custom crossover for variable-length kernel lists"""

        left1, right1 = self.decode_individual(ind1)
        left2, right2 = self.decode_individual(ind2)

        left1_head = left1[0]
        left2_head = left2[0]

        right1_head = right1[0]
        right2_head = right2[0]

        # Pick a favorite parent
        left_choice = random.choice([left1_head, left2_head])
        right_choice = random.choice([right1_head, right2_head])

        left_diff = abs(left1_head - left2_head)
        right_diff = abs(right1_head - right2_head)


        random_val = min( int(np.floor(np.abs(np.random.normal(loc=0, scale=4.12)))) , 10)
        percentage = random_val / 100.0

        left1[0] = max( abs(int(left_choice + (percentage * left_diff))), self.min_kernel_size )
        left2[0] = max( abs(int(left_choice - (percentage * left_diff))), self.min_kernel_size )

        right1[0] = max( abs(int(right_choice + (percentage * right_diff))), self.min_kernel_size )
        right2[0] = max( abs(int(right_choice - (percentage * right_diff))), self.min_kernel_size )

        # Reconstruct individuals
        ind1[:] = [len(left1)] + left1 + [len(right1)] + right1
        ind2[:] = [len(left2)] + left2 + [len(right2)] + right2

        return ind1, ind2

    def mutate(self, individual):
        """Custom mutation for kernel sizes and branch lengths"""
        left_kernels, right_kernels = self.decode_individual(individual)
        
        branch = random.choice(['left', 'right'])      
        target_branch = left_kernels if branch == 'left' else right_kernels
        
        # Mutate a kernel size value
        target_branch[0] = random.randint(self.min_kernel_size, self.max_kernel_size)

        # Reconstruct individual
        individual[:] = [len(left_kernels)] + left_kernels + [len(right_kernels)] + right_kernels
        
        return individual,
    
    def run_evolution(self):
        """Run the evolutionary algorithm"""
        if self.verbose:
            print(f"Starting evolution with {self.population_size} individuals for {self.generations} generations")
        
        # Create initial population
        population = self.toolbox.population(n=self.population_size)
        
        # Run evolution
        result_pop = ModifiedEASimple(
            population, 
            self.toolbox,
            cxpb=self.cx_prob,
            mutpb=self.mut_prob,
            ngen=self.generations,
            LogManager=self.LogManager,
            stats=self.stats,
            halloffame=self.hall_of_fame,
            verbose=self.verbose
        )
        
        return result_pop, self.hall_of_fame, self.stats
    
    def log_results(self):
        
        def get_hall_of_fame_format(i):
            individual = self.hall_of_fame[i]
            left, right = self.decode_individual(individual)
            return f"{i+1}. Left={left}, Right={right}, Fitness={individual.fitness.values[0]:.4f}"

        self.LogManager.log_experiment(
            sleepstage= self.sleepstage,
            signal_type= self.signal_type,
            max_kernel_size= self.max_kernel_size,
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
            left, right = self.decode_individual(individual)
            print(f"  {i+1}. Left={left}, Right={right}, Fitness={individual.fitness.values[0]:.4f}")