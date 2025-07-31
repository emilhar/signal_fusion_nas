import inspect
import sympy
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_target_objs = None

def get_stage_map(classification_class):
    if not _target_objs:
        print("No targets, failure")
        quit()

    stage_map = {}
    for target in _target_objs:
        stage_map[target.data_label] = 1 if classification_class.given_name == target.given_name else 0

    return stage_map

class EvolutionManager:

    # Overview Manager
    POPULATION_SIZE: int = 1
    GENERATIONS: int = 2
    SELECTION_TOURNAMENT_SIZE = max(3, int(POPULATION_SIZE * 0.2))
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
    VERBOSE = True
    VERY_VERBOSE = True # Shows Individual Training sessions

class DataManager:
    class DatasetNames:
        EDF_20 = "sleep-EDF-20"
        EDF_78 = "sleep-EDF-78"
        EDFx = "sleep_EDFx"

    _datasets = [DatasetNames.EDF_20, DatasetNames.EDF_78]

    DATASET = _datasets[1]
    MAX_MEMORY = 2048*2

    # SleepDataLoader
    dataset_percentage = 0.3

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

    MINIMIZE_FITNESS = True
    fitness_function = train_loss
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
    experiment_id = "experiment_id"
    fitness = "fitness"
    fitness_function = "fitness_function"
    generation = "generation"
    generations = "generations"
    hall_of_fame_members = "hall_of_fame_members"
    indi_id = "individual_id"
    learning_rate = "learning_rate"
    lr = "learning_rate"
    max_kernel_size = "max_kernel_size"
    max_memory = "max_memory"
    max_number_of_mutations = "max_number_of_mutations"
    min_kernel_size = "min_kernel_size"
    minimize_fitness = "minimize_fitness"
    mutation_prob = "mutation_prob"
    name = "name"
    number_of_branches_range = "number_of_branches_range"
    number_of_kernels_range = "number_of_kernels_range"
    population_size = "population_size"
    precision = "precision"
    reason = "reason"
    recall = "recall"
    second_best = "second_best"
    selection_tournament_size = "selection_tournament_size"
    signal_type = "signal_type"
    classification_class = "classification_class"
    start_time = "start_time"
    state_dict = "state_dict"
    test_loss = "test_loss"
    third_best = "third_best"
    time = "time"
    train_loss = "train_loss"

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
