from Globals import *
import argparse

def parse_arguments():
    """Parse command line arguments to override global settings"""
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--pop-size', type=int, help=f'Population size (default: {EvolutionManager.POPULATION_SIZE})')
    parser.add_argument('--generations', type=int, help=f'Number of generations (default: {EvolutionManager.GENERATIONS})')
    
    # New arguments for Globals class
    parser.add_argument('--binary-epochs', type=int, help=f'Epochs for fully training binary models (default: {Globals.epochs_for_fully_training_binary_models})')
    parser.add_argument('--ensemble-epochs', type=int, help=f'Epochs for training ensemble models (default: {Globals.epochs_for_training_ensemble_models})')
    parser.add_argument('--ea-datapoints', type=int, help=f'Number of datapoints per individual (default: {Globals.ea_datapoints_per_individual})')
    parser.add_argument('--max-filters', type=int, help=f'Maximum filters for Theseus (default: {Globals.max_filters_for_theseus})')
    parser.add_argument('--confusion-folder', type=str, help=f'Confusion matrix folder name (default: {Globals.confusion_matrix_folder_name})')
    parser.add_argument('--lazy-mem', type=int, help=f'Maximum memory for lazy data loader (default: {Globals.lazy_data_max_memory // 2**10}GiB)')

    args = parser.parse_args()
    apply_arguments(args)
    return args

def apply_arguments(args):
    """Apply parsed arguments to global settings"""
    # EvolutionManager settings
    if args.pop_size:
        EvolutionManager.POPULATION_SIZE = args.pop_size
    if args.generations:
        EvolutionManager.GENERATIONS = args.generations - 1
    
    # New Globals settings
    if args.binary_epochs:
        Globals.epochs_for_fully_training_binary_models = args.binary_epochs
    if args.ensemble_epochs:
        Globals.epochs_for_training_ensemble_models = args.ensemble_epochs
    if args.ea_datapoints:
        Globals.ea_datapoints_per_individual = args.ea_datapoints
    if args.max_filters:
        Globals.max_filters_for_theseus = args.max_filters
    if args.confusion_folder:
        Globals.confusion_matrix_folder_name = args.confusion_folder
    if args.lazy_mem:
        Globals.lazy_data_max_memory = args.lazy_mem
    