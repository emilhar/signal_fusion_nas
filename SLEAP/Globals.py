import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EvolutionManager:

    # Overview Manager
    POPULATION_SIZE: int = 2
    GENERATIONS: int = 1

    # Evolution Manager
    CX_PROB: float = 0.5
    MUTATION_PROB: float = 0.5

    # Misc
    VERBOSE = False
    VERY_VERBOSE = True and False # Shows Individual Training sessions

class LoggingHelper:
    LOGGING = False

    current_generation_id = 0
    current_individual_id = 0
    population_size = 0

    experiment_name = "Unnamed"
