"""
These names are used by many classes, good idea to keep them global
"""
import random
import inspect
import sympy

class Sleepstage:
    WAKE = "wake"
    N3 = "N3"
    N2 = "N2"
    N1 = "N1"
    REM = "REM"

    ALL_STAGES = [WAKE, N1, N2, N3, REM]

class Signal:

    class EEG:
        Fpz_Cz = "EEG_Fpz-Cz"
        Pz_Oz = "EEG_Pz-Oz"

    class EOG:
        HORIZONTAL = "EOG_horizontal"

    class EMG:
        SUBMENTAL = "EMG_submental"

    SIGNAL_COUNT = 3000
    ALL_SIGNALS = [EEG.Fpz_Cz, EEG.Pz_Oz, EOG.HORIZONTAL, EMG.SUBMENTAL]

class ModelManager:
    # Base
    NUMBER_OF_BRANCHES_RANGE = (1, 3)
    NUMBER_OF_KERNELS_RANGE = (2, 4)
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4

    # Kernel size constraints
    MIN_KERNEL_SIZE = 1
    MAX_KERNEL_SIZE = None

class EvolutionManager:

    # Overview Manager
    POPULATION_SIZE_PER_LAYER: int = 5
    GENERATIONS: int = 2
    SELECTION_TOURNAMENT_SIZE = max(3, int(POPULATION_SIZE_PER_LAYER * 0.2))
    ELITISM = 1
    HALL_OF_FAME_MEMBERS = 3

    MAX_NUMBER_OF_MUTATIONS = 3

    # Data split
    DATA_SPLIT_TRAINING = 0.7
    DATA_SPLIT_TESTING = 0.3
    VALID_DATA_SPLIT = (DATA_SPLIT_TRAINING + DATA_SPLIT_TESTING == 1)

    # Evolution Manager
    CX_PROB: float = 0.5
    MUTATION_PROB: float = 0.5

    # Misc
    VERBOSE = False # Shows Layers
    VERY_VERBOSE = False # Shows Individual Training sessions
    
class TimeWall:
    """After FLIP_ON% of generations, 
    Time Wall cuts out the worst TIME_WALL_PERCENTAGE% of performers when it comes to training time."""

    # After what % of generations do you turn on the time wall?
    FLIP_ON = 0.5
    ON = False

    STARTING_PERCENTAGE = 0.25
    MAX_PERCENTAGE = 0.75
    INCREASE = 0.05
    
    time_wall_percentage = 0.0

class AlpsManager:
    AGE_GAP = 2

    class AgingScheme:
        FIBBONACCI = [1, 2, 3, 5, 8, 13, 21, "NA"]
        LINEAR = [1, 2, 3, 4, 5, 6, "NA"]

        teitur = [1, 5, "NA"]
    
        used_aging_scheme = teitur
        uas_str = "Linear"

    MAX_AGE_IN_LAYERS = []
    for x in AgingScheme.used_aging_scheme:
        MAX_AGE_IN_LAYERS.append(x * AGE_GAP if isinstance(x, int) else x)


    # Create layers just before individuals try to move into them
    LAYER_CREATION_THRESHOLDS = [max_age for max_age in MAX_AGE_IN_LAYERS if not isinstance(max_age, str)]
    created_layers = [0]

    REAL_PERCENTAGES = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 1.00]
    TEST_PERCENTAGES = [0.05, 0.05, 0.05]
    teitur_percentages = [0.10, 0.20, 1.00]

    percentages = teitur_percentages
    # TRAINING_SETTINGS_FOR_LAYERS = None
    TRAINING_SETTINGS_FOR_LAYERS = {
        0: {"dataset_percentage": percentages[0], "training_epochs": 1,  "batch_size": ModelManager.BATCH_SIZE,    "learning_rate":ModelManager.LEARNING_RATE, "mu": EvolutionManager.POPULATION_SIZE_PER_LAYER, "lambda_": EvolutionManager.POPULATION_SIZE_PER_LAYER//2},
        1: {"dataset_percentage": percentages[1], "training_epochs": 3,  "batch_size": ModelManager.BATCH_SIZE,    "learning_rate":ModelManager.LEARNING_RATE, "mu": EvolutionManager.POPULATION_SIZE_PER_LAYER, "lambda_": EvolutionManager.POPULATION_SIZE_PER_LAYER//2},
        2: {"dataset_percentage": percentages[2], "training_epochs": 10,  "batch_size": ModelManager.BATCH_SIZE,     "learning_rate":ModelManager.LEARNING_RATE, "mu": EvolutionManager.POPULATION_SIZE_PER_LAYER, "lambda_": EvolutionManager.POPULATION_SIZE_PER_LAYER//2},
    }

    def _get_manager(percentages):
        return {
        0: {"dataset_percentage": percentages[0], "training_epochs": 1,  "batch_size": ModelManager.BATCH_SIZE,    "learning_rate":ModelManager.LEARNING_RATE, "mu": EvolutionManager.POPULATION_SIZE_PER_LAYER, "lambda_": EvolutionManager.POPULATION_SIZE_PER_LAYER//2},
        1: {"dataset_percentage": percentages[1], "training_epochs": 1,  "batch_size": ModelManager.BATCH_SIZE,    "learning_rate":ModelManager.LEARNING_RATE, "mu": EvolutionManager.POPULATION_SIZE_PER_LAYER, "lambda_": EvolutionManager.POPULATION_SIZE_PER_LAYER//2},
        2: {"dataset_percentage": percentages[2], "training_epochs": 1,  "batch_size": ModelManager.BATCH_SIZE,    "learning_rate":ModelManager.LEARNING_RATE, "mu": EvolutionManager.POPULATION_SIZE_PER_LAYER, "lambda_": EvolutionManager.POPULATION_SIZE_PER_LAYER//2},
        #3: {"dataset_percentage": percentages[3], "training_epochs": 3,  "batch_size": ModelManager.BATCH_SIZE,    "learning_rate":ModelManager.LEARNING_RATE, "mu": EvolutionManager.POPULATION_SIZE_PER_LAYER, "lambda_": EvolutionManager.POPULATION_SIZE_PER_LAYER//2},
        #4: {"dataset_percentage": percentages[4], "training_epochs": 10,  "batch_size": ModelManager.BATCH_SIZE,     "learning_rate":ModelManager.LEARNING_RATE, "mu": 5, "lambda_": 1},
        # 3: {"dataset_percentage": percentages[3], "training_epochs": 10,  "batch_size": ModelManager.BATCH_SIZE * 4,     "learning_rate":ModelManager.LEARNING_RATE, "mu": 5, "lambda_": 1},
        # 4: {"dataset_percentage": percentages[4], "training_epochs": epochs[4],  "batch_size": ModelManager.BATCH_SIZE * 2, "learning_rate":ModelManager.LEARNING_RATE, "mu": EvolutionManager.POPULATION_SIZE_PER_LAYER, "lambda_": EvolutionManager.POPULATION_SIZE_PER_LAYER//2},
        # 5: {"dataset_percentage": percentages[5], "training_epochs": epochs[5],  "batch_size": ModelManager.BATCH_SIZE * 2, "learning_rate":ModelManager.LEARNING_RATE, "mu": EvolutionManager.POPULATION_SIZE_PER_LAYER, "lambda_": EvolutionManager.POPULATION_SIZE_PER_LAYER//2},
        # 6: {"dataset_percentage": percentages[6], "training_epochs": epochs[6], "batch_size": ModelManager.BATCH_SIZE * 4, "learning_rate":ModelManager.LEARNING_RATE, "mu": 5, "lambda_": 1},
    }
    
class PolyarithmosManager:
    folder_path = None
    
class DataManager:
    class DatasetNames:
        EDF_20 = "sleep-EDF-20"
        EDF_78 = "sleep-EDF-78"
        EDFx = "sleep_EDFx"

    _datasets = [DatasetNames.EDF_20, DatasetNames.EDF_78]

    DATASET = _datasets[0]
    MAX_MEMORY = 256
    EVEN_DATA_SPLIT = False

class LoggingSettings:
    LOG_IDS = ['O', 'T']
    LOGGER_ID = ""
    LOGGING = True
    LOG_ALL_INDIVIDUALS = False

    current_generation_id = 0
    current_individual_id = 0
    population_size = 0

    experiment_name = "Unnamed"

class FitnessFunctions:
    @staticmethod
    def f1(individual_performance):
        fitness = individual_performance.get(LoggingTemplate.best_f1, 0.0)
        return fitness
    
    @staticmethod
    def train_loss(individual_performance):
        fitness = individual_performance.get(LoggingTemplate.train_loss, 0.0)
        return fitness
    
    @staticmethod
    def random_fitness(individual_performance):
        return random.random()

    @staticmethod
    def prime_fitness(branches):
        return sum(item for branch in branches for item in branch if sympy.isprime(item))
    
    @staticmethod
    def closeness_to_global_opt(branches):
        global_optimum = [[19, 18], [420, 120, 8], [1000, 1000, 1000]]
        
        # Sort each branch in both the input and global optimum
        sorted_branches = sorted(branches, key=lambda x: len(x))
        
        score = 0
        
        # Length mismatch penalty
        if len(sorted_branches) != len(global_optimum):
            score -= 10_000
            return score  # Early return for severe mismatch
        
        # Compare each corresponding branch
        for branch, optimum_branch in zip(sorted_branches, global_optimum):
            # Length mismatch within branch
            if len(branch) != len(optimum_branch):
                score -= 1_000
                continue
                
            # Calculate element-wise distance (using Manhattan distance)
            for a, b in zip(branch, optimum_branch):
                score -= abs(a - b)  # Negative because lower distance is better
                
        return score
    
    @staticmethod
    def train_loss_normalize(individual, population):

        losses = [x.fitness.values[0] for x in population]
        highest_loss_val = max(losses)
        lowest_loss_val = min(losses)
        loss = individual.fitness.values[0]

        if highest_loss_val == lowest_loss_val:
            fitness = 1.0
        else:
            fitness = (highest_loss_val - loss) / (highest_loss_val - lowest_loss_val)

        individual.fitness.values = (fitness,)

    @staticmethod
    def no_normalization(individual, population):
        pass

    MINIMIZE_FITNESS = False
    fitness_function = closeness_to_global_opt
    normalization_function = no_normalization


class Clr:
    def __init__(self, string, color=None, bg_color=None):
        self.string = str(string)
        self.color = color.lower() if color else None
        self.bg_color = bg_color.lower() if bg_color else None
    
    def __str__(self):
        # Foreground colors
        fg_colors = {
            "black": "\033[30m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            "bright_black": "\033[90m",
            "bright_red": "\033[91m",
            "bright_green": "\033[92m",
            "bright_yellow": "\033[93m",
            "bright_blue": "\033[94m",
            "bright_magenta": "\033[95m",
            "bright_cyan": "\033[96m",
            "bright_white": "\033[97m",
        }
        
        # Background colors
        bg_colors = {
            "black": "\033[40m",
            "red": "\033[41m",
            "green": "\033[42m",
            "yellow": "\033[43m",
            "blue": "\033[44m",
            "magenta": "\033[45m",
            "cyan": "\033[46m",
            "white": "\033[47m",
            "bright_black": "\033[100m",
            "bright_red": "\033[101m",
            "bright_green": "\033[102m",
            "bright_yellow": "\033[103m",
            "bright_blue": "\033[104m",
            "bright_magenta": "\033[105m",
            "bright_cyan": "\033[106m",
            "bright_white": "\033[107m",
        }
        
        codes = []
        
        # Add foreground color if specified
        if self.color and self.color in fg_colors:
            codes.append(fg_colors[self.color])
        
        # Add background color if specified
        if self.bg_color and self.bg_color in bg_colors:
            codes.append(bg_colors[self.bg_color])
        
        
        reset_code = "\033[0m"
        color_code = "".join(codes)
        
        return f"{color_code}{self.string}{reset_code}"

class LoggingTemplate:
    rounding_number = 2

    accuracy = "accuracy"
    age = "age"
    age_gap = "age_gap"
    aging_scheme = "aging_scheme"
    alps_manager = "alps_manager"
    base_batch_size = "base_batch_size"
    best = "best"
    best_auc = "best_auc"
    best_f1 = "best_f1"
    best_scores = "best_scores"
    best_true = "true_labels"
    branches = "branches"
    crossover_prob = "crossover_prob"
    data_split_testing = "data_split_testing"
    data_split_training = "data_split_training"
    dataset_name = "dataset_name"
    elitism = "elitism"
    end_time = "end_time"
    epoch = "epoch"
    even_data_split = "even_data_split"
    experiment_id = "experiment_id"
    experiment_ids_within_polyarithmos = "experiment_ids_within_polyartihmos"
    fitness = "fitness"
    fitness_function = "fitness_function"
    generation = "generation"
    generations = "generations"
    hall_of_fame_members = "hall_of_fame_members"
    indi_id = "individual_id"
    layer = "layer"
    layer_creation_thresholds = "layer_creation_thresholds"
    learning_rate = "learning_rate"
    lr = "learning_rate"
    max_age_in_layers = "max_age_in_layers"
    max_kernel_size = "max_kernel_size"
    max_memory = "max_memory"
    max_number_of_mutations = "max_number_of_mutations"
    min_kernel_size = "min_kernel_size"
    minimize_fitness = "minimize_fitness"
    mutation_prob = "mutation_prob"
    name = "name"
    number_of_branches_range = "number_of_branches_range"
    number_of_kernels_range = "number_of_kernels_range"
    p_type = "p_type"
    percentages = "percentages"
    population_size = "population_size"
    precision = "precision"
    reason = "reason"
    recall = "recall"
    rounding_number = 2
    second_best = "second_best"
    selection_tournament_size = "selection_tournament_size"
    signal_type = "signal_type"
    sleepstage = "sleepstage"
    start_time = "start_time"
    state_dict = "state_dict"
    test_loss = "test_loss"
    third_best = "third_best"
    time = "time"
    time_wall_flip_on = "time_wall_flip_on"
    time_wall_increase = "time_wall_increase"
    time_wall_max_percentage = "time_wall_max_percentage"
    time_wall_starting_percentage = "time_wall_starting_percentage"
    train_loss = "train_loss"

    # Translation table
    reason_map = {
        "Best In Layer": 0,
        "Checked For Best In Generation": 1
    }

class SLEAPyException(Exception):
    def __init__(self, **kwargs):
        super().__init__()
        print(f"{Clr("All classes and their contents in Globals.py", "blue")}:")

        for k, v in kwargs.items():
            print(f"{Clr(k, "red")}: {Clr(v, "red")}")

        # Collect all top-level classes in the module
        module_classes = {
            name: obj for name, obj in globals().items()
            if inspect.isclass(obj) and obj.__module__ == __name__
        }

        for class_name, cls in module_classes.items():
            print(f"\nClass: {class_name}")
            for attr_name, attr_value in inspect.getmembers(cls):
                if attr_name.startswith("__") and attr_name.endswith("__"):
                    continue  # Skip dunder methods

                if inspect.isfunction(attr_value):
                    print(f"  Method: {attr_name}()")
                elif isinstance(attr_value, (int, float, str, list, dict, tuple, bool)):
                    print(f"  Variable: {attr_name} = {repr(attr_value)}")
                elif inspect.isclass(attr_value):
                    print(f"  Nested Class: {attr_name}")
