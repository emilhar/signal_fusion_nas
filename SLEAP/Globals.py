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
    NUMBER_OF_BRANCHES_RANGE = (1, 4)
    NUMBER_OF_KERNELS_RANGE = (1, 3)
    BATCH_SIZE = 32
    TRAINING_EPOCHS_PER_INDIVIDUAL: int = 2
    MAX_TIME_SPENT_TRAINING = 3
    LEARNING_RATE = 5e-5

    # Kernel size constraints
    MIN_KERNEL_SIZE = 1
    MAX_KERNEL_SIZE = None

    # Misc
    SMALLER_FILES = False
    VERBOSE = True

class EvolutionSettings:

    # Overview settings
    POPULATION_SIZE: int = 10
    GENERATIONS: int = 20
    SELECTION_TOURNAMENT_SIZE = 5
    HALL_OF_FAME_MEMBERS = 3

    MAX_NUMBER_OF_MUTATIONS = 3

    # Data split
    DATA_POINTS_PER_INDIVIUAL = 4300
    DATA_SPLIT_TRAINING = 0.7
    DATA_SPLIT_TESTING = 0.3
    VALID_DATA_SPLIT = (DATA_SPLIT_TRAINING + DATA_SPLIT_TESTING == 1)

    # Fitness Settings:
    # alpha and beta are used in the fitness function
    #   alpha is how much you value fitness score
    #   beta is how much you value uniqueness
    # these values change over time as generations come and go

    ALPHA_BETA = [0.7, 0.3]
    alpha = ALPHA_BETA [0]
    beta = ALPHA_BETA[1]
    BETA_SWITCH = 1/2

    # Evolution settings
    CX_PROB: float = 0.7
    MUTATION_PROB: float = 0.4

    # Full training settings
    KOTH_ON = True
    KOTH_GENERATIONS_BETWEEN = 1600000
    KOTH_TOURNAMENT_SIZE = 0.30
    FULL_TRAIN_BATCH_SIZE = 128
    FULL_TRAIN_EPOCHS = 20
    FULL_TRAIN_LEARNING_RATE_MULTIPLIER = 10

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

    experiment_name = "My lovely experiment"

class UniquenessFunctions:

    @staticmethod
    def gargoyle(individual, comparisons):
        if not comparisons:
            if ModelSettings.VERBOSE:
                print(f"Missing comparisons, giving full uniqueness score to {individual}")
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
        transition = ModelSettings.MAX_KERNEL_SIZE / pow(EvolutionSettings.POPULATION_SIZE, 1/3)

        print(min_distance)

        return 1 / (1 + math.exp(-steepness * (min_distance - transition)))

    uniqueness_function = gargoyle


class FitnessFunctions:
    @staticmethod
    def f1(individual_performance):
        fitness = individual_performance.get("F1", 0.0)
        return fitness
    
    @staticmethod
    def train_loss(individual_performance):
        fitness = individual_performance.get("Train Loss", 0.0)
        return fitness
    
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

        return fitness
    
    @staticmethod
    def branch_size(individual_performance):
        branches = individual_performance.get("Branches", [[]])
        
        if sum(branches) < 1000:
            return 1.0
        else:
            return 0.0

    fitness_function = train_loss
    normalize = (True, train_loss_normalize)

