"""
These names are used by many classes, good idea to keep them global
"""
# Available batch sizes for all models
BATCH_SIZE_OPTIONS = [2, 4, 8, 16, 32, 64, 128]

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

    EMG_SIGNAL_COUNT = 30
    NON_EMG_SIGNAL_COUNT = 3000

class ModelSettings:
    # Base
    NUMBER_OF_BRANCHES = 1
    BATCH_SIZE = 32  # Default batch size (from BATCH_SIZE_OPTIONS)
    TRAINING_EPOCHS_PER_INDIVIDUAL: int = 1
    KERNELS_PER_BRANCH = 3
    VERBOSE = True
    MAX_TIME_SPENT_TRAINING = 3
    LEARNING_RATE = 5e-5

    # Kernel size constraints
    SORT_KERNELS = True
    MIN_KERNEL_SIZE = 1
    MAX_KERNEL_SIZE = None

    # Misc
    SMALLER_FILES = False

class EvolutionSettings:

    # Overview settings
    POPULATION_SIZE: int = 20
    GENERATIONS: int = 10
    SELECTION_TOURNAMENT_SIZE = 5
    HALL_OF_FAME_MEMBERS = 3
    # 'F1' or 'F1 + Unique'
    FITNESS_FUNCTION = "F1 + Unique"

    # Data split
    DATA_POINTS_PER_INDIVIUAL = 4300
    DATA_SPLIT_TRAINING = 0.7
    DATA_SPLIT_TESTING = 0.3
    VALID_DATA_SPLIT = (DATA_SPLIT_TRAINING + DATA_SPLIT_TESTING == 1)

    # Fitness Settings:
    alpha = 0
    beta = 1
    ufunctions = ["manhattan distance", "reverse manhattan distance"]
    uniqueness_function = ufunctions[0]

    # Evolution settings
    CX_PROB: float = 0.7
    MUTATION_PROB: float = 0.4
    OFFSPRING_VARIATION = 5     # When crossover happens, how different are the children from their parents?
    LAYERS_OF_CNN = 3


    # Tournament of Champion settings
    TDB_ON = True
    TDB_GENERATIONS_BETWEEN = 16
    TDB_TOURNAMENT_SIZE = 0.30
    TDB_BATCH_SIZE = 128
    TDB_EPOCHS = 2
    TDB_LEARNING_RATE_MULTIPLIER = 10

class DataSettings:
    class DatasetNames:
        TELEMETRY = "telemetry"
        SLEEPEDFX = "sleepEDFX"

    _datasets = [DatasetNames.SLEEPEDFX, DatasetNames.TELEMETRY]
    DATASET = _datasets[1]

class LoggingSettings:
    LOG_IDS = ['O', 'T']
    LOGGER_ID = ""
    LOGGING = True
    LOG_INDIVIDUALS = True # Champions always get logged


class UniquenessFunctions:
    @staticmethod
    def manhattan_distance(individual, comparisons):
        if not comparisons:
            return 1

        def distance(a, b):
            return sum(abs(x - y) for x, y in zip(a, b))
    
        max_possible_dist = ModelSettings.KERNELS_PER_BRANCH * (
                ModelSettings.MAX_KERNEL_SIZE - ModelSettings.MIN_KERNEL_SIZE
            )

        uniqueness_scores = [
            distance(individual[0], other[0])
            for other in comparisons
        ]

        avg_distance = sum(uniqueness_scores) / len(uniqueness_scores)
        
        # Normalize uniqueness to [0,1]
        uniqueness = avg_distance / max_possible_dist if max_possible_dist > 0 else 0.0
        return uniqueness

    @staticmethod
    def _reverse_manhattan_distance(individual, comparisons):

        max_possible_dist = ModelSettings.KERNELS_PER_BRANCH * (
                ModelSettings.MAX_KERNEL_SIZE - ModelSettings.MIN_KERNEL_SIZE
            )

        def distance(a, b):
            dist = max_possible_dist - sum(abs(x - y) for x, y in zip(a, b))
            return dist / max_possible_dist

        uniqueness_scores = [
            distance(individual[0], other[0])
            for other in comparisons
        ]
        sum_denominator = sum(uniqueness_scores)

        uniqueness = 1 / (1+sum_denominator)
        return uniqueness

    @staticmethod
    def punishing_reverse_manhattan(individual, comparisons):
        if not comparisons:
            return 1.0

        max_possible_dist = ModelSettings.KERNELS_PER_BRANCH * (
            ModelSettings.MAX_KERNEL_SIZE - ModelSettings.MIN_KERNEL_SIZE
        )

        def normalized_distance(a, b):
            dist = sum(abs(x - y) for x, y in zip(a, b))
            return dist / max_possible_dist

        # Use inverse-square to punish close neighbors harshly
        uniqueness_scores = [
            1 / (normalized_distance(individual[0], other[0]) + 1e-6)**2  # avoid div by 0
            for other in comparisons
        ]

        avg_inverse_penalty = sum(uniqueness_scores) / len(uniqueness_scores)

        # Invert so that large penalties -> low uniqueness
        uniqueness = 1 / (1 + avg_inverse_penalty)
        return uniqueness
    
    @staticmethod
    def KILL(individual, comparisons):

        max_possible_dist = ModelSettings.KERNELS_PER_BRANCH * (
                ModelSettings.MAX_KERNEL_SIZE - ModelSettings.MIN_KERNEL_SIZE
            )

        def distance(a, b):
            dist = max_possible_dist - sum(abs(x - y) for x, y in zip(a, b))
            return dist / max_possible_dist

        uniqueness_scores = [
            distance(individual[0], other[0])
            for other in comparisons
        ]
        sum_denominator = sum(uniqueness_scores)

        uniqueness = 1 / (1/sum_denominator) if sum_denominator != 0 else 0.0
        return uniqueness


    uniqueness_function = punishing_reverse_manhattan

