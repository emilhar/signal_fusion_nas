import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DATASET_PATH = "/mnt/datasets/shhs_preprocessed_chunks"
DATASET_PATH = "/mnt/datasets/fake_data"


class EvolutionManager:

    # Overview Manager
    POPULATION_SIZE: int = 2
    GENERATIONS: int = 1

    # Evolution Manager
    CX_PROB: float = 0.5
    MUTATION_PROB: float = 0.5


class Globals:
    class GigaBytes:
        G_1 = 2**10
        G_2 = (2**10) * (2)
        G_4 = (2**10) * (2**2)
        G_8 = (2**10) * (2**3)
        G_16 = (2**10) * (2**4)

        G_36 = 2*G_16 + G_4

    epochs_for_fully_training_binary_models = 1
    epochs_for_training_ensemble_models = 1

    ea_datapoints_per_individual = 5000
    max_filters = 32
    confusion_matrix_folder_name = "clash_royale"
    lazy_data_max_memory = GigaBytes.G_16 * 2

    max_processes = 1

