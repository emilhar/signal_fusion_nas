from Globals import *
import argparse

def parse_arguments():
    """Parse command line arguments to override global settings"""
    parser = argparse.ArgumentParser(description='SLEAPy - Sleep Labeling using Evolutionary Algorithms and PyTorch')
    
    # ModelManager options
    parser.add_argument('--batch-size', type=int, help=f'Batch size (default: {ModelManager.BATCH_SIZE})')
    parser.add_argument('--lr', type=float, help=f'Learning rate (default: {ModelManager.LEARNING_RATE})')
    parser.add_argument('--min-ks', type=int, help=f'Minimum kernel size (default: {ModelManager.MIN_KERNEL_SIZE})')
    parser.add_argument('--max-ks', type=int, help=f'Maximum kernel size (default: {ModelManager.MAX_KERNEL_SIZE})')
    
    # EvolutionManager options
    parser.add_argument('--pop-size', type=int, help=f'Population size (default: {EvolutionManager.POPULATION_SIZE})')
    parser.add_argument('--generations', type=int, help=f'Number of generations (default: {EvolutionManager.GENERATIONS})')
    parser.add_argument('--st-size', type=int, help=f'Selection tournament size (default: {EvolutionManager.SELECTION_TOURNAMENT_SIZE})')
    parser.add_argument('--hof-size', type=int, help=f'Hall of fame size (default: {EvolutionManager.HALL_OF_FAME_MEMBERS})')
    parser.add_argument('--cx-prob', type=float, help=f'Crossover probability (default: {EvolutionManager.CX_PROB})')
    parser.add_argument('--mut-prob', type=float, help=f'Mutation probability (default: {EvolutionManager.MUTATION_PROB})')
    parser.add_argument('--verbose', action='store_true', help=f'Verbose output (default: {EvolutionManager.VERBOSE})')
    parser.add_argument('--v-verbose', action='store_true', help=f'Prints individual training sessions, (default {EvolutionManager.VERY_VERBOSE})' )

    # DataManager options
    parser.add_argument('--dataset', choices=['sleep-EDF-20', 'sleep-EDF-78', 'sleep-EDFx'], help=f'Dataset to use (default: {DataManager.DATASET})')
    parser.add_argument('--max-mem', type=int, help=f'Maximum memory for lazyloader cache (default: {DataManager.MAX_MEMORY})')
    parser.add_argument('--even-split', action='store_true', help=f'Use even data  (default: {DataManager.EVEN_DATA_SPLIT})')
    
    # LoggingSettings options
    parser.add_argument('--no-logging', action='store_true', help='Disable logging')
    parser.add_argument('--log', action='store_true', help='Enabme logging')
    parser.add_argument('--log-id', choices=LoggingSettings.LOG_IDS, help=f'Logger ID (default: {LoggingSettings.LOGGER_ID})')
    parser.add_argument('--log-all', action='store_true', help=f'Log all individuals (default: {LoggingSettings.LOG_ALL_INDIVIDUALS})')
    parser.add_argument('--exp-name', type=str, help=f'Experiment name (default: {LoggingSettings.experiment_name})')
    
    parser.add_argument('--sleep-stage', type=str, choices=[s for s in Sleepstage.ALL_STAGES],
                       help='Sleep stage to analyze (wake, N1, N2, N3, REM)')
    parser.add_argument('--signal', type=str, 
                       choices=['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG_horizontal', 'EMG_submental'],
                       help='Signal type to use')

    parser.add_argument('--polyarithmos', action='store_true', help='Run all possible configurations (ultimate test mode)')

    args = parser.parse_args()
    apply_arguments(args)

    return args

def apply_arguments(args):
    """Apply parsed arguments to global settings"""
    # ModelManager settings
    if args.batch_size:
        ModelManager.BATCH_SIZE = args.batch_size
    if args.lr:
        ModelManager.LEARNING_RATE = args.learning_rate
    if args.min_ks:
        ModelManager.MIN_KERNEL_SIZE = args.min_ks
    if args.max_ks:
        ModelManager.MAX_KERNEL_SIZE = args.max_ks
    
    # EvolutionManager settings
    if args.pop_size:
        EvolutionManager.POPULATION_SIZE = args.pop_size
        # Update tournament size if it's based on population size
        EvolutionManager.SELECTION_TOURNAMENT_SIZE = max(3, int(EvolutionManager.POPULATION_SIZE * 0.2))
    if args.generations:
        EvolutionManager.GENERATIONS = args.generations - 1
    if args.st_size:
        EvolutionManager.SELECTION_TOURNAMENT_SIZE = args.tournament_size
    if args.hof_size:
        EvolutionManager.HALL_OF_FAME_MEMBERS = args.hof_size
    if args.cx_prob:
        EvolutionManager.CX_PROB = args.cx_prob
    if args.mut_prob:
        EvolutionManager.MUTATION_PROB = args.mut_prob
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
    if args.even_split:
        DataManager.EVEN_DATA_SPLIT = True
    
    # LoggingSettings settings
    if args.no_logging:
        LoggingSettings.LOGGING = False
    if args.log:
        LoggingSettings.LOGGING = True
    if args.log_id:
        LoggingSettings.LOGGER_ID = args.log_id
    if args.log_all:
        LoggingSettings.LOG_ALL_INDIVIDUALS = True
    if args.exp_name:
        LoggingSettings.experiment_name = args.exp_name