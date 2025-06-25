import random
import numpy as np
from deap import base, creator, tools
from EAController.SleepDataLoader import SleepDataLoader
from math import sqrt

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
                 
                 # Evolution parameters
                 population_size: int = EvolutionSettings.POPULATION_SIZE,
                 generations: int = EvolutionSettings.GENERATIONS,
                 cx_prob: float = EvolutionSettings.CX_PROB,
                 mut_prob: float = EvolutionSettings.MUTATION_PROB,
                 tournament_size: int = EvolutionSettings.SELECTION_TOURNAMENT_SIZE,
                 
                 # Kernel size constraints
                 min_kernel_size: int = ModelSettings.MIN_KERNEL_SIZE,
                 max_kernel_size: int|None = ModelSettings.MAX_KERNEL_SIZE,
                 
                 verbose: bool = ModelSettings.VERBOSE):
        
        # Base
        self.sleepstage = sleepstage
        self.signal_type = signal_type
        self.batch_size = batch_size
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
        """Generate a random individual with kernel branches"""

        branches = []

        for _ in range( ModelSettings.NUMBER_OF_BRANCHES ):

            kernel =  [random.randint(self.min_kernel_size, self.max_kernel_size) for _ in range(ModelSettings.KERNELS_PER_BRANCH)]
            if ModelSettings.SORT_KERNELS:
                kernel.sort(reverse=True)
            branches.append(kernel)

        # Individual format: [[branch1_kernels], [branch2_kernels], ..., [branchN_kernels]]

        individual = []

        for branch in branches:
            individual.append(branch)

        return creator.Individual(individual)

    def evaluate_individual(self, individual, champion=False):
        """Evaluate an individual by training a model
        arg: individual
        arg: champion Bool, if individual is partaking in a tournament of champions, then they train on the full dataset"""
        
        # Train model and get performance
        model_performance = self.create_individual(individual, champion)
        
        fitness_value = self.calculate_fitness(individual, model_performance)
        
        if self.verbose:
            print(f"Fitness: {fitness_value}")

        if LoggingSettings.LOGGING:

            train_loss = model_performance.get("Train Loss", 0.0),
            test_loss = model_performance.get("Test Loss", 0.0),
            precision = model_performance.get("Precision", 0.0),
            recall = model_performance.get("Recall", 0.0),
            f1 = model_performance.get("F1", 0.0),
            accuracy = model_performance.get("Accuracy", 0.0),

            self.LogManager.check_for_best_in_gen(individual, fitness_value, champion, train_loss, test_loss, precision, recall, f1, accuracy)
        
        return (fitness_value,)
    
    def create_individual(self, branches: list[list[int]], champion=False):

        if champion:
            individual_training_set, individual_test_set, n_samples, pos_weight = self.SDL.get_full_dataset()
            batch_size = EvolutionSettings.TOC_BATCH_SIZE
            epochs = EvolutionSettings.TOC_EPOCHS
            learning_rate = ModelSettings.LEARNING_RATE * EvolutionSettings.TOC_LEARNING_RATE_MULTIPLIER
        else:
            individual_training_set, individual_test_set, n_samples, pos_weight = self.SDL.get_random_subset() 
            batch_size = self.batch_size
            epochs = self.epochs
            learning_rate = ModelSettings.LEARNING_RATE


        # Things marked with # come from the SDL
        new_model = TrainedModelMaker(
            branches = branches,
            name=f"{branches}, sleepstage: {self.sleepstage}, {batch_size}batch, {self.epochs}epochs",
            sleepstage = self.sleepstage,
            signal_type=self.signal_type,
            batch_size= batch_size,
            train_loader = individual_training_set,
            test_loader = individual_test_set,
            epochs= epochs,
            learning_rate=learning_rate,
            verbose= self.verbose,
            N_SAMPLES= n_samples, #
            pos_weight= pos_weight,
            champion=champion) #

        return new_model.model_performance

    def calculate_fitness(self, individual, model_performance):
        if EvolutionSettings.FITNESS_FUNCTION == "F1":
            f1_score = model_performance.get("F1", 0.0)
            return f1_score

        elif EvolutionSettings.FITNESS_FUNCTION == "F1 + Unique":
            f1_score = model_performance.get("F1", 0.0)
            current = individual[0]

            population = [ind[0] for ind in self.toolbox.population(n=self.population_size)]

            # Compute average distance to others
            distances = [
                self._distance(current, other)
                for other in population
                if other != current
            ]
            uniqueness = sum(distances) / len(distances) if distances else 0.0
            uniqueness /= self.max_kernel_size * sqrt(ModelSettings.KERNELS_PER_BRANCH)

            print(round(uniqueness, 5))

            return EvolutionSettings.alpha * f1_score + EvolutionSettings.beta * uniqueness

        else:
            raise ValueError("No valid fitness function chosen")

    def _distance(self, a, b):
                return sum(abs(x - y) for x, y in zip(a, b))
    
    def crossover(self, ind1, ind2):
        """Custom crossover for variable-length kernel lists with multiple branches"""

        num_branches = len(ind1)
        assert len(ind2) == num_branches, "Both individuals must have the same number of branches"

        for i in range(num_branches):
            branch1 = ind1[i]
            branch2 = ind2[i]

            if not branch1 or not branch2:
                continue  # Skip empty branches

            # Choose "head" values
            head1 = branch1[0]
            head2 = branch2[0]

            favorite = random.choice([head1, head2])
            diff = abs(head1 - head2)

            # Gaussian noise
            random_val = min(int(np.floor(abs(np.random.normal(loc=0, scale=4.12)))), 10)
            percentage = random_val / 100.0

            new_head1 = max(int(favorite + percentage * diff), self.min_kernel_size)
            new_head2 = max(int(favorite - percentage * diff), self.min_kernel_size)

            branch1[0] = new_head1
            branch2[0] = new_head2

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
                new_value = max(self.min_kernel_size, min(self.max_kernel_size, new_value))
                branch[i] = new_value

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
            return f"{i+1}. Branches={individual}, Fitness={individual.fitness.values[0]:.4f}"

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
            print(f"  {i+1}. Branches={individual}, Fitness={individual.fitness.values[0]:.4f}")
