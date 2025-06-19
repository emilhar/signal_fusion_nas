"""
These names are used by many classes, good idea to keep them global
"""
# Available batch sizes for all models
BATCH_SIZE_OPTIONS = [2, 4, 8, 16, 32, 64, 128]

class Sleepstage:
    WAKE = "wake"
    LIGHT_SLEEP = "light-sleep"
    DEEP_SLEEP = "deep-sleep"
    REM = "REM"

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
    BATCH_SIZE = 32  # Default batch size (from BATCH_SIZE_OPTIONS)
    TRAINING_EPOCHS_PER_INDIVIDUAL: int = 4
    DATASET_FRACTION: float = 0.3
    VERBOSE = True
    MAX_TIME_SPENT_TRAINING = 3

    # Base kernel sizes
    KERNEL = [400, 8, 8]

    # Kernel size constraints
    MIN_KERNEL_SIZE = 1
    MAX_KERNEL_SIZE = None

    # Misc
    SMALLER_FILES = False

class EvolutionSettings:

    # Overview settings
    POPULATION_SIZE: int = 20
    GENERATIONS: int = 15
    TOURNAMENT_SIZE = 3
    HALL_OF_FAME_MEMBERS = 3
    LOGGING = True

    # Individual settings
    DATA_POINTS_PER_INDIVIUAL = 3000
    CX_PROB: float = 0.7
    MUTATION_PROB: float = 0.2
    OFFSPRING_VARIATION = 3     # When crossover happens, how different are the children from their parents?
    LAYERS_OF_CNN = 3
    RANDOM_KERNELS_PER_BRANCH = 1

    # Tournament of Champion settings
    TOC_ON = True
    TOC_GENERATIONS_BETWEEN = 5
    TOC_TOURNAMENT_SIZE = 0.20
    TOC_BATCH_SIZE = 128  # Tournament batch size (from BATCH_SIZE_OPTIONS)

class DataSettings:
    class DatasetNames:
        TELEMETRY = "telemetry"
        SLEEPEDFX = "sleepEDFX"

    _datasets = ["sleepEDFX", "telnet"]
    DATASET = _datasets[0]

class LoggingSettings:
    FITNESS_NEEDED_FOR_LOG = 0.40
    LOG_INDIVIDUALS = False # Champions always get logged