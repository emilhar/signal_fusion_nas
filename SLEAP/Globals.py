import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_PATH = "/mnt/datasets/shhs_preprocessed_chunks"

class EvolutionManager:

    # Overview Manager
    POPULATION_SIZE: int = 100
    GENERATIONS: int = 10

    # Evolution Manager
    CX_PROB: float = 0.5
    MUTATION_PROB: float = 0.5

class Globals:
    epochs_for_fully_training_binary_models = 30
    epochs_for_training_ensemble_models = 10

    ea_datapoints_per_individual = 5000
    max_filters = 32
    confusion_matrix_folder_name = "clash_royale"
    lazy_data_max_memory = (2**10) * (2**4)

