from Globals import *
from ea_controller.ea_controller import KernelSizeEvolutionaryOptimizer
import argparse

def parse_arguments():
    """Parse command line arguments to override global settings"""
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--min-ks', type=int, help=f'Minimum kernel size (default: {KernelSizeEvolutionaryOptimizer.MIN_KERNEL_SIZE})')
    parser.add_argument('--max-ks', type=int, help=f'Maximum kernel size (default: {KernelSizeEvolutionaryOptimizer.MAX_KERNEL_SIZE})')
    
    parser.add_argument('--pop-size', type=int, help=f'Population size (default: {EvolutionManager.POPULATION_SIZE})')
    parser.add_argument('--generations', type=int, help=f'Number of generations (default: {EvolutionManager.GENERATIONS})')
    parser.add_argument('--verbose', action='store_true', help=f'Verbose output (default: {EvolutionManager.VERBOSE})')
    parser.add_argument('--v-verbose', action='store_true', help=f'Prints individual training sessions, (default {EvolutionManager.VERY_VERBOSE})' )

    parser.add_argument('--dataset', choices=['sleep-EDF-20', 'sleep-EDF-78', 'sleep-EDFx'], help=f'Dataset to use (default: {DataManager.DATASET})')
    parser.add_argument('--max-mem', type=int, help=f'Maximum memory for lazyloader cache (default: {DataManager.MAX_MEMORY})')
    
    parser.add_argument('--no-logging', action='store_true', help='Disable logging')    

    args = parser.parse_args()
    apply_arguments(args)

    return args

def apply_arguments(args):
    """Apply parsed arguments to global settings"""
    if args.min_ks:
        KernelSizeEvolutionaryOptimizer.MIN_KERNEL_SIZE = args.min_ks
    if args.max_ks:
        KernelSizeEvolutionaryOptimizer.MAX_KERNEL_SIZE = args.max_ks
    
    # EvolutionManager settings
    if args.pop_size:
        EvolutionManager.POPULATION_SIZE = args.pop_size
        # Update tournament size if it's based on population size
        EvolutionManager.SELECTION_TOURNAMENT_SIZE = max(3, int(EvolutionManager.POPULATION_SIZE * 0.2))
    if args.generations:
        EvolutionManager.GENERATIONS = args.generations - 1
    if args.st_size:
        EvolutionManager.SELECTION_TOURNAMENT_SIZE = args.st_size
    if args.hof_size:
        EvolutionManager.HALL_OF_FAME_MEMBERS = args.hof_size
    if args.verbose:
        EvolutionManager.VERBOSE = True
    if args.v_verbose:
        EvolutionManager.VERBOSE = True
        EvolutionManager.VERY_VERBOSE = True
    
    # DataManager settings
    if args.dataset:
        DataManager.DATASET = args.dataset
    if args.max_mem:
        DataManager.MAX_MEMORY = args.max_mem
    
    # LoggingSettings settings
    if args.no_logging:
        LoggingSettings.LOGGING = False
    if args.log:
        LoggingSettings.LOGGING = True
    if args.log_id:
        LoggingSettings.LOGGING = True
        LoggingSettings.LOGGER_ID = args.log_id
    if args.log_all:
        LoggingSettings.LOGGING = True
        LoggingSettings.LOG_ALL_INDIVIDUALS = True
    if args.exp_name:
        LoggingSettings.LOGGING = True
        LoggingSettings.experiment_name = args.exp_name