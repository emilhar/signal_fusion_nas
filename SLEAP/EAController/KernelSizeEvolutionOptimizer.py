import random
import numpy as np
from deap import base, creator, tools
from EAController.SleepDataLoader import SleepDataLoader

from ModelController.TrainedModelMaker import TrainedModelMaker
from Globals import Signal, ModelSettings, EvolutionSettings, AlpsSettings, LoggingSettings, UniquenessFunctions, FitnessFunctions

from EAController.SLeaMuPlusLambda import SLeaMuPlusLambda
from Logs.LogManager import LogManager


"""""
Færa fyrst, svo búa til nýtt layer
"""""

class KernelSizeEvolutionaryOptimizer:

    def __init__(self, sleepstage: str, signal_type: str):
        
        # Base
        self.sleepstage = sleepstage
        self.signal_type = signal_type

        if ModelSettings.MAX_KERNEL_SIZE == None:
            ModelSettings.MAX_KERNEL_SIZE = self.find_max_kernel_size()
            if ModelSettings.VERBOSE: print(f"Max kernel size set at {ModelSettings.MAX_KERNEL_SIZE}")
        
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
        self.toolbox.register("select", self.select, tournsize=EvolutionSettings.SELECTION_TOURNAMENT_SIZE, elitism=EvolutionSettings.ELITISM)
        
        # Genetic operatorss
        self.toolbox.register("evaluate", self.evaluate_individual)
        
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
            first = max(1, random.choice( range(ModelSettings.MIN_KERNEL_SIZE, ModelSettings.MAX_KERNEL_SIZE, 20) ) - 1)
            branch = [first]
            
            for _ in range(kernel_per_branch[i]-1):
                item  = max( branch[-1] // 2, 1)
                branch.append(item)
                
            branches.append(branch)

        # Individual format: [[branch1_kernels], [branch2_kernels], ..., [branchN_kernels]]

        individual = creator.Individual(branches)
        individual.raw_fitness = None
        individual.uniqueness = None
        individual.age =  0
        individual.layer = 0

        if EvolutionSettings.beta <= 0:
            individual.alpha_beta_fitness = None

        return individual
    
    def evaluate_individual(self, individual):
        """Evaluate an individual by training a model
        arg: individual"""

        # If the individual has passed the maximum age in their layer, then they train as if they are in the layer above.
        # This is then used in the comparison later.
        if individual.age > AlpsSettings.MAX_AGE_IN_LAYERS[individual.layer]:
            fake_layer = individual.layer + 1
            model_performance = self.create_trained_individual(individual, fake_layer)
        else:
            model_performance = self.create_trained_individual(individual, individual.layer)

        raw_fitness = self.calculate_fitness(model_performance)

        individual.model_performance = model_performance
        individual.raw_fitness = raw_fitness
        individual.individual_id = LoggingSettings.current_individual_id
        LoggingSettings.current_individual_id += 1

        return (raw_fitness,)
    
    def create_trained_individual(self, individual, layer):
        """Creates trained individuals. Is used to create all individuals who aren't in the first-generation"""

        time_limit = False

        individual_training_set, individual_test_set, n_samples, pos_weight = self.SDL.get_random_subset(
            dataset_percentage = AlpsSettings.TRAINING_SETTINGS_FOR_LAYERS[layer]["dataset_percentage"],
            batch_size=AlpsSettings.TRAINING_SETTINGS_FOR_LAYERS[layer]["batch_size"]) 
        
        batch_size = AlpsSettings.TRAINING_SETTINGS_FOR_LAYERS[layer]["batch_size"]
        epochs =AlpsSettings.TRAINING_SETTINGS_FOR_LAYERS[layer]["training_epochs"]
        learning_rate = ModelSettings.LEARNING_RATE

        new_model = TrainedModelMaker(
            branches = individual,
            name=f"{individual}, {batch_size}batch, {ModelSettings.TRAINING_EPOCHS_PER_INDIVIDUAL}epochs",
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
            have_time_limit = time_limit
        )

        return new_model.model_performance

    def calculate_fitness(self, model_performance):
        return FitnessFunctions.fitness_function(model_performance)

    def select(self, population, number_of_people_to_select, tournsize, elitism):
        """Tournament selection"""

        if number_of_people_to_select <= 0:
            return []
        
        map(lambda x: FitnessFunctions.normalization_function(x, population), population)
        min_or_max = min if FitnessFunctions.MINIMIZE_FITNESS else max
        
        self.chosen_for_next_generation = []
        
        for _ in range(number_of_people_to_select):
            aspirants = [random.choice(population) for _ in range(tournsize)]
            chosen = min_or_max(aspirants, key=lambda x: self._selection_criteria(x))
            self.chosen_for_next_generation.append( chosen )

        if LoggingSettings.LOGGING:
            for individual in population:
                self.LogManager.check_for_best_in_gen(individual)
        
        return self.chosen_for_next_generation

    def _selection_criteria(self, individual):
        if self.chosen_for_next_generation == []:
            return individual.fitness.values[0]
        
        uniqueness = UniquenessFunctions.uniqueness_function(individual, self.chosen_for_next_generation)

        alpha_beta_fitness = (
            EvolutionSettings.alpha * individual.fitness.values[0] + 
            EvolutionSettings.beta * uniqueness
        )

        individual.uniqueness = uniqueness
        individual.alpha_beta_fitness = alpha_beta_fitness
        
        return alpha_beta_fitness

    def crossover(self, ind1, ind2):
        """Custom crossover for variable-length kernel lists with multiple branches"""

        branch_count_ind1 = len(ind1)
        branch_count_ind2 = len(ind2)

        # Both individuals have >1 branch
        if ( (branch_count_ind1 > 1) and (branch_count_ind2 > 1) ):
            children = self._large_branch_crossover(ind1, ind2)
            
        # Both individuals have exactly 1 branch
        elif ( (branch_count_ind1 == 1) and (branch_count_ind2 == 1) ):
            children = self._small_branch_crossover(ind1, ind2)

        # Exactly one individual has 1 branch while other has >1 branches
        else:
            if branch_count_ind1 == 1:
                single_branch = ind1
                other = ind2

            elif branch_count_ind2 == 1:
                single_branch = ind2
                other = ind1

            children =  self._mixed_branch_crossover(single_branch, other)

        return children

    def _large_branch_crossover(self, ind1, ind2):
        """Crossover function for individuals when BOTH individuals have more than one branch.
        Individuals trade a single branch"""

        # Decide branch to trade
        range_max = min(len(ind1), len(ind2)) - 1
        trade_branch_index = random.randrange(range_max)

        # Trade branches
        ind1[trade_branch_index], ind2[trade_branch_index] = ind2[trade_branch_index], ind1[trade_branch_index]

        return ind1, ind2

    def _mixed_branch_crossover(self, single_branch, other):
        """Crossover function for when one individual has exactly 1 branch, but the other has many branches.
        A single branch is chosen from the larger individual, then small_branch_crossover is performed on those branches"""
        branch_choice_index = random.randrange(len(other)-1)

        child1, child2 = self._small_branch_crossover(single_branch, [other[branch_choice_index]])

        # Update the original individuals with the two new, varied branches
        single_branch[0] = child1[0]
        other[branch_choice_index] = child2[0]

        return single_branch, other

    def _small_branch_crossover(self, ind1, ind2):
        """Crossover function for when both individuals have exactly 1 branch. 
        Performs one-point crossover on the single branch from two individuals."""
        branch1 = ind1[0]
        branch2 = ind2[0]

        if min(len(branch1), len(branch2))-1 < 1:
            picked = max([ind1, ind2], key=lambda x: len(x[0]))
            return picked, picked
            
        
        cx_point = random.randint(1, min(len(branch1), len(branch2))-1)
        child_branch1 = branch1[:cx_point] + branch2[cx_point:]
        child_branch2 = branch2[:cx_point] + branch1[cx_point:]

        ind1[0] = sorted(child_branch1, reverse=True)
        ind2[0] = sorted(child_branch2, reverse=True)

        return ind1, ind2

    def mutate(self, individual):
        """Mutate an individual by randomly modifying branches or kernel sizes."""
        mutant_age = individual.age
        
        # Clone the individual to avoid in-place issues
        mutant = creator.Individual([branch[:] for branch in individual])
        mutant.age = mutant_age

        mutation_number_options = [i + 1 for i in range(EvolutionSettings.MAX_NUMBER_OF_MUTATIONS)
              for _ in range(EvolutionSettings.MAX_NUMBER_OF_MUTATIONS - i)]
        
        number_of_mutations = random.choice(mutation_number_options)

        for _ in range(number_of_mutations):
            mutation_type = self._mutation_choice()
        
            if mutation_type == "add_branch":
                if len(mutant) < ModelSettings.NUMBER_OF_BRANCHES_RANGE[1]:
                    branch_length = random.randint(*ModelSettings.NUMBER_OF_KERNELS_RANGE)
                    first_kernel = max(1, random.choice(range(ModelSettings.MIN_KERNEL_SIZE, ModelSettings.MAX_KERNEL_SIZE, 20))-1)
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

        return (mutant,)

    def _mutation_choice(self):
        """
        Randomly selects a mutation type based on predefined probability ranges.
        """
        num = random.randint(0, 99)
        mutation_options = ["remove_branch", "add_branch", "add_kernel", "remove_kernel", "change_kernel"]

        if 0 <= num < 10:
            return mutation_options[0]  # remove_branch (10%)
        elif 10 <= num < 25:
            return mutation_options[1]  # add_branch (15%)
        elif 25 <= num < 50:
            return mutation_options[2]  # add_kernel (25%)
        elif 50 <= num < 75:
            return mutation_options[3]  # remove_kernel (25%)
        else:
            return mutation_options[4]  # change_kernel (25%)

    def run_evolution(self):
        """Run the evolutionary algorithm"""
        if ModelSettings.VERBOSE:
            print(f"Starting evolution with {EvolutionSettings.POPULATION_SIZE_PER_LAYER} individuals for {EvolutionSettings.GENERATIONS} generations")

        # Create initial population
        population = self.toolbox.population(n=EvolutionSettings.POPULATION_SIZE_PER_LAYER)
        
        # Run evolution
        evolver = SLeaMuPlusLambda(
            population=population,
            toolbox=self.toolbox
        )
        
        result_pop = evolver.main(
            cxpb= EvolutionSettings.CX_PROB,
            mutpb= EvolutionSettings.MUTATION_PROB,
            ngen= EvolutionSettings.GENERATIONS,
            LogManager= self.LogManager,
            stats= self.stats,
            halloffame= self.hall_of_fame,
            verbose= ModelSettings.VERBOSE

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
