import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EvolutionManager:

    # Overview Manager
    POPULATION_SIZE: int = 15
    GENERATIONS: int = 5

    # Evolution Manager
    CX_PROB: float = 0.5
    MUTATION_PROB: float = 0.5

    # Misc
    VERBOSE = False
    VERY_VERBOSE = True and False # Shows Individual Training sessions


class Globals:
    epochs_for_fully_training_binary_models = 20
    epochs_for_training_ensemble_models = 10

    ea_datapoints_per_individual = 5000
    max_filters_for_theseus = 4
    confusion_matrix_folder_name = "Debug"
    lazy_data_max_memory = (2**10) * (2**5)

class LoggingHelper:
    LOGGING = False

    current_generation_id = 0
    current_individual_id = 0
    population_size = 0

    experiment_name = "Unnamed"

    CONF_SAVE_DIR = None
