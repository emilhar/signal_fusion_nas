import random
import numpy as np
from deap import base, creator, tools
from EAController.SleepDataLoader import SleepDataLoader

from ModelController.TrainedModelMaker import TrainedModelMaker
from Globals import Signal, ModelManager, EvolutionManager, AlpsManager, LoggingManager, FitnessFunctions

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

        if LoggingManager.LOGGING:
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

        for i in range(number_of_branches):
            first = max(1, random.choice( range(ModelManager.MIN_KERNEL_SIZE, ModelManager.MAX_KERNEL_SIZE, 20) ) - 1)
            branch = [first]
            
            for _ in range(kernel_per_branch[i]-1):
                item  = max( branch[-1] // 2, 1)
                branch.append(item)
                
            branches.append(branch)

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
        individual.individual_id = LoggingManager.current_individual_id
        LoggingManager.current_individual_id += 1

        return (fitness,)
    
    def create_trained_individual(self, individual, layer):
        """Creates trained individuals. Is used to create all individuals who aren't in the first-generation"""

        time_limit = ModelManager.HAVE_MAX_TIME

        individual_training_set, individual_test_set, n_samples, pos_weight = self.SDL.get_random_subset(
            dataset_percentage = AlpsManager.TRAINING_Manager_FOR_LAYERS[layer]["dataset_percentage"],
            batch_size=AlpsManager.TRAINING_Manager_FOR_LAYERS[layer]["batch_size"]) 
        
        batch_size = AlpsManager.TRAINING_Manager_FOR_LAYERS[layer]["batch_size"]
        epochs =AlpsManager.TRAINING_Manager_FOR_LAYERS[layer]["training_epochs"]
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
            verbose= EvolutionManager.VERBOSE,
            N_SAMPLES= n_samples,
            pos_weight= pos_weight,
            have_time_limit = time_limit
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
        min_or_max = min if FitnessFunctions.MINIMIZE_FITNESS else max

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
        
        # Perform tournament selection for remaining individuals
        for _ in range(remaining_to_select):
            aspirants = [random.choice(population) for _ in range(tournsize)]
            chosen = min_or_max(aspirants, key=lambda x: x.fitness.values[0])
            self.chosen_for_next_generation.append(chosen)

        if LoggingManager.LOGGING:
            for individual in population:
                self.LogManager.check_for_best_in_gen(individual)
        
        return self.chosen_for_next_generation

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

        ind1[0] = child_branch1
        ind2[0] = child_branch2

        return ind1, ind2

    def mutate(self, individual):
        """Mutate an individual by randomly modifying branches or kernel sizes."""
        
        # Clone the individual to avoid in-place issues
        mutant = creator.Individual([branch[:] for branch in individual])
        mutant.age = individual.age
        mutant.layer = individual.layer

        mutation_number_options = [i + 1 for i in range(EvolutionManager.MAX_NUMBER_OF_MUTATIONS)
              for _ in range(EvolutionManager.MAX_NUMBER_OF_MUTATIONS - i)]
        
        number_of_mutations = random.choice(mutation_number_options)

        for _ in range(number_of_mutations):
            mutation_type = self._mutation_choice()
        
            if mutation_type == "add_branch":
                if len(mutant) < ModelManager.NUMBER_OF_BRANCHES_RANGE[1]:
                    branch_length = random.randint(*ModelManager.NUMBER_OF_KERNELS_RANGE)
                    first_kernel = max(1, random.choice(range(ModelManager.MIN_KERNEL_SIZE, ModelManager.MAX_KERNEL_SIZE, 20))-1)
                    new_branch = [first_kernel]
                    for _ in range(branch_length - 1):
                        new_branch.append(max( new_branch[-1] // 2, 1))
                    mutant.append(new_branch)

            elif mutation_type == "remove_branch":
                if len(mutant) > ModelManager.NUMBER_OF_BRANCHES_RANGE[0]:
                    mutant.pop(random.randrange(len(mutant)))

            elif mutation_type == "add_kernel":
                branch_idx = random.randrange(len(mutant))
                if len(mutant[branch_idx]) < ModelManager.NUMBER_OF_KERNELS_RANGE[1]:
                    new_kernel = max(ModelManager.MIN_KERNEL_SIZE, mutant[branch_idx][-1] // 2)
                    mutant[branch_idx].append(new_kernel)

            elif mutation_type == "remove_kernel":
                branch_idx = random.randrange(len(mutant))
                if len(mutant[branch_idx]) > ModelManager.NUMBER_OF_KERNELS_RANGE[0]:
                    mutant[branch_idx].pop(random.randrange(len(mutant[branch_idx])))

            elif mutation_type == "change_kernel":
                branch_idx = random.randrange(len(mutant))
                kernel_idx = random.randrange(len(mutant[branch_idx]))
                current_value = mutant[branch_idx][kernel_idx]
                change = random.choice([-100, -50, 50, 100])
                new_value = min(max(current_value + change, ModelManager.MIN_KERNEL_SIZE), ModelManager.MAX_KERNEL_SIZE)
                mutant[branch_idx][kernel_idx] = new_value

            elif mutation_type == "randomize_kernel_order_in_branch":
                branch_idx = random.randrange(len(mutant))
                random.shuffle(mutant[branch_idx])

        return (mutant,)

    def _mutation_choice(self):
        """
        Randomly selects a mutation type based on predefined probability ranges.
        """
        num = random.randint(0, 99)

        if ModelManager.SORT_KERNELS:
            # Sorted kernels
            if 0 <= num <= 14:       # 15%: Remove branch
                return "remove_branch"
            elif 15 <= num <= 29:    # 15%: Add branch
                return "add_branch"
            elif 30 <= num <= 54:    # 25%: Add kernel
                return "add_kernel"
            elif 55 <= num <= 69:    # 15%: Remove kernel
                return "remove_kernel"
            else:                    # 30%: Change kernel
                return "change_kernel"

        else:
            # Unsorted kernels (unsort added)
            if 0 <= num <= 14:       # 15%: Remove branch
                return "remove_branch"
            elif 15 <= num <= 29:    # 15%: Add branch
                return "add_branch"
            elif 30 <= num <= 49:    # 20%: Add kernel
                return "add_kernel"
            elif 50 <= num <= 59:    # 10%: Remove kernel
                return "remove_kernel"
            elif 60 <= num <= 84:    # 25%: Change kernel
                return "change_kernel"
            else:                    # 15%: Unsort
                return "randomize_kernel_order_in_branch"

    def run_evolution(self):
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
        
        return result_pop, self.hall_of_fame, self.stats

    def log_results(self):
        
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
    
    def print_results(self):
        """Print evolution results"""
        print("\n" + "="*50)
        print("EVOLUTION RESULTS")
        print("="*50)
        
        print(f"\nHall of Fame (Top {len(self.hall_of_fame)}):")
        for i, individual in enumerate(self.hall_of_fame):
            print(f"  {i+1}. Branches={individual}, Fitness={individual.fitness.values[0]:.4f}")
