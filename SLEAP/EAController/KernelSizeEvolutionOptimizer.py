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
        """Generate a random individual with kernel branches.
        Only used to create the first generation of individuals"""

        branches = []

        number_of_branches = random.randint( ModelSettings.NUMBER_OF_BRANCHES_RANGE[0], ModelSettings.NUMBER_OF_BRANCHES_RANGE[1])
        kernel_per_branch = [ random.randint( ModelSettings.NUMBER_OF_KERNELS_RANGE[0], ModelSettings.NUMBER_OF_KERNELS_RANGE[1]) for _ in range(number_of_branches) ]

        for i in range(number_of_branches):

            first = max(
                1,
                random.choice( range(ModelSettings.MIN_KERNEL_SIZE, ModelSettings.MAX_KERNEL_SIZE, 20)) - 1)
            branch = [first]

            for _ in range(kernel_per_branch[i]-1):
                item  = max( branch[-1] // 2, 1)
                branch.append(item)
                
            branches.append(branch)

        # Individual format: [[branch1_kernels], [branch2_kernels], ..., [branchN_kernels]]

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
        individual.individual_id = LoggingSettings.current_individual_id

        LoggingSettings.current_individual_id += 1


        if ModelSettings.VERBOSE:
            print(f"Fitness: {fitness_value}")

        return (fitness_value,)
    
    def create_trained_individual(self, branches: list[list[int]], full_training=False):
        """Creates trained individuals. Is used to create all individuals who aren't in the first-generation"""

        if full_training:
            individual_training_set, individual_test_set, n_samples, pos_weight = self.SDL.get_full_dataset()
            batch_size = EvolutionSettings.FULL_TRAIN_BATCH_SIZE
            epochs = EvolutionSettings.FULL_TRAIN_EPOCHS
            learning_rate = ModelSettings.LEARNING_RATE * EvolutionSettings.FULL_TRAIN_LEARNING_RATE_MULTIPLIER
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

    def select(self, population, k, tournsize):
        
        if EvolutionSettings.CX_PROB == 0.0 and EvolutionSettings.MUTATION_PROB == 0.0:
            if LoggingSettings.LOGGING:
                for individual in population:
                    self.LogManager.check_for_best_in_gen(individual)

            return population
        
        self.chosen = []
        for _ in range(k):
            aspirants = [random.choice(population) for _ in range(tournsize)]
            next_up = max(aspirants, key=lambda x: self._selection_criteria(x, population))
            self.chosen.append(next_up)

        if LoggingSettings.LOGGING:
            for individual in population:
                self.LogManager.check_for_best_in_gen(individual)

        return self.chosen

    def _selection_criteria(self, individual, population):

        if FitnessFunctions.normalize[0] == True:
            fitness = FitnessFunctions.normalize[1](individual, population)

        to_be_compared = [ind for ind in self.chosen if ind != individual]
        
        uniqueness = UniquenessFunctions.uniqueness_function(individual, to_be_compared)

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

        branch_count_ind1 = len(ind1)
        branch_count_ind2 = len(ind2)

        # Both individuals have >1 branch
        if ( (branch_count_ind1 > 1) and (branch_count_ind2 > 1) ):
            return self._large_branch_crossover(ind1, ind2)
            
        # Both individuals have exactly 1 branch
        elif ( (branch_count_ind1 == 1) and (branch_count_ind2 == 1) ):
            return self._small_branch_crossover(ind1, ind2)

        # Exactly one individual has 1 branch while other has >1 branches
        if branch_count_ind1 == 1:
            single_branch = ind1
            other = ind2

        elif branch_count_ind2 == 1:
            single_branch = ind2
            other = ind1

        return self._medium_branch_crossover(single_branch, other)

    def _large_branch_crossover(self, ind1, ind2):
        """Crossover function for individuals when BOTH individuals have more than one branch.
        Individuals trade a single branch"""

        # Decide branch to trade
        range_max = min(len(ind1), len(ind2)) - 1
        trade_branch_index = random.randrange(range_max)

        # Trade branches
        ind1[trade_branch_index], ind2[trade_branch_index] = ind2[trade_branch_index], ind1[trade_branch_index]

        return ind1, ind2

    def _medium_branch_crossover(self, single_branch, other):
        """Crossover function for when one individual has exactly 1 branch, but the other has many branches.
        A single branch is chosen from the larger individual, then small_branch_crossover is performed on those branches"""
        branch_choice_index = random.randrange(len(other)-1)

        other[branch_choice_index], single_branch[0] = self._small_branch_crossover(single_branch[0], other[branch_choice_index])

        return single_branch, other
        
    def _small_branch_crossover(self, ind1, ind2):
        """Crossover function for when both individuals have exactly 1 branch.
        Picks one branch"""
        picked = random.choice([ind1, ind2])
    
        return picked, picked

    def mutate(self, individual):
        """Mutate an individual by randomly modifying branches or kernel sizes."""
        
        # Clone the individual to avoid in-place issues
        mutant = creator.Individual([branch[:] for branch in individual])
        
        number_of_mutations = random.randint(0, EvolutionSettings.MAX_NUMBER_OF_MUTATIONS)
        mutation_types = ["add_branch", "remove_branch", "add_kernel", "remove_kernel", "change_kernel"]

        for _ in range(number_of_mutations):
            mutation_type = random.choice(mutation_types)
        
            if mutation_type == "add_branch":
                if len(mutant) < ModelSettings.NUMBER_OF_BRANCHES_RANGE[1]:
                    branch_length = random.randint(*ModelSettings.NUMBER_OF_KERNELS_RANGE)
                    first_kernel = max(1, random.choice(range(ModelSettings.MIN_KERNEL_SIZE, ModelSettings.MAX_KERNEL_SIZE, 20) - 1))
                    new_branch = [first_kernel]
                    for _ in range(branch_length - 1):
                        new_branch.append(max( new_branch[-1] // 2, 1))
                    mutant.append(new_branch)

            elif mutation_type == "remove_branch":
                if len(mutant) > ModelSettings.NUMBER_OF_BRANCHES_RANGE[0]:
                    mutant.pop(random.randrange(len(mutant)))

            elif mutation_type == "add_kernel":
                branch_idx = random.randrange(len(mutant))
                if len(mutant[branch_idx]) < ModelSettings.NUMBER_OF_KERNELS_RANGE[1]:
                    new_kernel = max(ModelSettings.MIN_KERNEL_SIZE, mutant[branch_idx][-1] // 2)
                    mutant[branch_idx].append(new_kernel)

            elif mutation_type == "remove_kernel":
                branch_idx = random.randrange(len(mutant))
                if len(mutant[branch_idx]) > ModelSettings.NUMBER_OF_KERNELS_RANGE[0]:
                    mutant[branch_idx].pop(random.randrange(len(mutant[branch_idx])))

            elif mutation_type == "change_kernel":
                branch_idx = random.randrange(len(mutant))
                kernel_idx = random.randrange(len(mutant[branch_idx]))
                current_value = mutant[branch_idx][kernel_idx]
                change = random.choice([-100, -50, 50, 100])
                new_value = min(max(current_value + change, ModelSettings.MIN_KERNEL_SIZE), ModelSettings.MAX_KERNEL_SIZE)
                mutant[branch_idx][kernel_idx] = new_value

        mutant.raw_fitness = None
        mutant.uniqueness = None
        mutant.alpha_beta_fitness = None
        mutant.fully_trained = False

        return (mutant,)

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
            