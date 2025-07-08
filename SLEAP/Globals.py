"""
These names are used by many classes, good idea to keep them global
"""

import math
import copy

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

class ModelSettings:
    # Base
    NUMBER_OF_BRANCHES_RANGE = (1, 3)
    NUMBER_OF_KERNELS_RANGE = (2, 4)
    BATCH_SIZE = 32
    TRAINING_EPOCHS_PER_INDIVIDUAL: int = 2
    MAX_TIME_SPENT_TRAINING = 6
    LEARNING_RATE = 5e-4

    # Kernel size constraints
    MIN_KERNEL_SIZE = 1
    MAX_KERNEL_SIZE = 50

    # Misc
    SMALLER_FILES = False
    VERBOSE = True

class EvolutionSettings:

    # Overview settings
    POPULATION_SIZE_PER_LAYER: int = 10
    GENERATIONS: int = 20
    SELECTION_TOURNAMENT_SIZE = max(3, int(POPULATION_SIZE_PER_LAYER * 0.2))
    ELITISM = 3
    HALL_OF_FAME_MEMBERS = 3

    MAX_NUMBER_OF_MUTATIONS = 3

    # Data split
    DATA_SPLIT_TRAINING = 0.7
    DATA_SPLIT_TESTING = 0.3
    VALID_DATA_SPLIT = (DATA_SPLIT_TRAINING + DATA_SPLIT_TESTING == 1)

    # Fitness Settings:
    # alpha and beta are used in the fitness function
    #   alpha is how much you value fitness score
    #   beta is how much you value uniqueness
    # these values change over time as generations come and go

    ALPHA_BETA = [1.0, 0.0]
    alpha = ALPHA_BETA [0]
    beta = ALPHA_BETA[1]
    BETA_SWITCH = 1/2           # NEVER HAPPENS

    # Evolution settings
    CX_PROB: float = 0.5
    MUTATION_PROB: float = 0.5
    
class AlpsSettings:
    AGE_GAP = 1

    class AgingScheme:
        FIBBONACCI = [1, 2, 3, 5, 8, 13, 21]
        LINEAR = [1, 2, 3, 4, 5, 6]
        

    MAX_AGE_IN_LAYERS = []
    for x in AgingScheme.FIBBONACCI:
        MAX_AGE_IN_LAYERS.append(x * AGE_GAP)


    # Create layers just before individuals try to move into them
    LAYER_CREATION_THRESHOLDS = [max_age for max_age in MAX_AGE_IN_LAYERS]

    DATASET_PERCENTAGES = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 1.00]
    TEST_PERCENTAGES = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

    percentages = TEST_PERCENTAGES
    TRAINING_SETTINGS_FOR_LAYERS = {
        0: {"dataset_percentage": percentages[0],  "training_epochs": 1,  "batch_size": ModelSettings.BATCH_SIZE,    "learning_rate":ModelSettings.LEARNING_RATE},
        1: {"dataset_percentage": percentages[1],  "training_epochs": 2,  "batch_size": ModelSettings.BATCH_SIZE,    "learning_rate":ModelSettings.LEARNING_RATE},
        2: {"dataset_percentage": percentages[2], "training_epochs": 3,  "batch_size": ModelSettings.BATCH_SIZE,     "learning_rate":ModelSettings.LEARNING_RATE},
        3: {"dataset_percentage": percentages[3], "training_epochs": 4,  "batch_size": ModelSettings.BATCH_SIZE,     "learning_rate":ModelSettings.LEARNING_RATE},
        4: {"dataset_percentage": percentages[4],  "training_epochs": 5,  "batch_size": ModelSettings.BATCH_SIZE * 2, "learning_rate":ModelSettings.LEARNING_RATE},
        5: {"dataset_percentage": percentages[5],  "training_epochs": 6,  "batch_size": ModelSettings.BATCH_SIZE * 2, "learning_rate":ModelSettings.LEARNING_RATE},
        6: {"dataset_percentage": percentages[6],  "training_epochs": 10, "batch_size": ModelSettings.BATCH_SIZE * 4, "learning_rate":ModelSettings.LEARNING_RATE},
    }

    individuals_and_fitnesses_in_layers = {}
    
class DataSettings:
    class DatasetNames:
        TELEMETRY = "telemetry"
        SLEEPEDFX = "sleepEDFX"

    _datasets = [DatasetNames.SLEEPEDFX, DatasetNames.TELEMETRY]
    DATASET = _datasets[1]

    EVEN_DATA_SPLIT = False

class LoggingSettings:
    LOG_IDS = ['O', 'T']
    LOGGER_ID = ""
    LOGGING = True
    LOG_ALL_INDIVIDUALS = True

    current_generation_id = 0
    current_individual_id = 0
    population_size = 0

    experiment_name = "Unnamed"

class UniquenessFunctions:

    @staticmethod
    def gargoyle(individual, comparisons):
        if not comparisons:
            return 1.0

        min_distance = float("inf")
        sorted_individual_copy = sorted(copy.deepcopy(individual), key=lambda x: len(x))

        for other in comparisons:
            sorted_other_copy = sorted(copy.deepcopy(other), key=lambda x: len(x))
            max_branch_count = max(len(sorted_individual_copy), len(sorted_other_copy))

            total_distance = 0

            for i in range(max_branch_count):
                # Get branches or empty list if not present
                branch_a = sorted_individual_copy[i] if i < len(sorted_individual_copy) else []
                branch_b = sorted_other_copy[i] if i < len(sorted_other_copy) else []

                max_len = max(len(branch_a), len(branch_b))

                # Pad with zeros
                padded_a = branch_a + [0] * (max_len - len(branch_a))
                padded_b = branch_b + [0] * (max_len - len(branch_b))

                # Euclidean distance between the padded branches
                total_distance += sum((a - b) ** 2 for a, b in zip(padded_a, padded_b))

            dist = math.sqrt(total_distance)

            if dist < min_distance:
                min_distance = dist

        steepness = 0.01
        transition = ModelSettings.MAX_KERNEL_SIZE / pow(EvolutionSettings.POPULATION_SIZE_PER_LAYER, 1/3)

        return 1 / (1 + math.exp(-steepness * (min_distance - transition)))

    uniqueness_function = gargoyle

class FitnessFunctions:
    @staticmethod
    def f1(individual_performance):
        fitness = individual_performance.get("F1", 0.0)
        return fitness
    
    @staticmethod
    def train_loss(individual_performance):
        raw_fitness = individual_performance.get("Train Loss", 0.0)
        return raw_fitness
    
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
    def train_loss_and_time(individual_performance):
        TL = individual_performance.get("Train Loss", 0.0)
        time = individual_performance.get("Time", 0.0) * 0.1

        return TL + time
        
    
    @staticmethod
    def no_normalization(individual, population):
        pass

    MINIMIZE_FITNESS = True
    fitness_function = train_loss
    normalization_function = no_normalization

import inspect

class SLEAP_Exception(Exception):
    def __init__(self, **kwargs):
        super().__init__()
        print("All classes and their contents in Globals.py:")

        for k, v in kwargs.items():
            print(f"{k}: {v}")

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
