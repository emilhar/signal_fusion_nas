"""
Gives IO for SLEAPy
"""
import argparse
import os
import torch

from Globals import Sleepstage, Signal, SLEAPyException
from EAController.KernelSizeEvolutionOptimizer import KernelSizeEvolutionaryOptimizer
from Logs.LogManager import LogManager
from Globals import ModelManager, EvolutionManager, DataManager, LoggingSettings, FitnessFunctions, AlpsManager, TimeWall, LoggingTemplate, PolyarithmosManager

class SLEAPy:
    """
    Sleep
    Labeling using
    Evolutionary
    Algorithms and
    Pytorch

    Main interface for running evolutionary optimization to find optimal kernel sizes
    """

    def __init__(self, args=None):
        self.optimizer = None
        self.sleepstage = None
        self.signal_type = None
        self.args = args
        
    def run_experiment(self, run_every_possible_experiment=False, minimax=False):
        """Run the setup and evolution process"""
        print("\n" + "="*68)
        print("🧠 SLEAPy - Sleep Labeling using Evolutionary Algorithms and PyTorch")
        print("="*68)
        
        if run_every_possible_experiment:
            self._run_every_possible_experiment()
          
        elif minimax:
            self._run_minimax()

        else:
            self._get_user_configuration()
            self._create_optimizer()
            self.optimizer.run_evolution()
    
    def _run_every_possible_experiment(self):
        print("\n🔥 ULTIMATE TEST MODE: Running all possible configurations")
        configs = self._generate_all_configs()
        self.sleepstage = "All sleep stages"
        self.signal_type = "All signal types"

        if LoggingSettings.LOGGING:
            while True:
                print("\n",LoggingSettings.LOG_IDS)
                potential_log_id = input("Enter logging ID: ").upper().strip()
                if potential_log_id in LoggingSettings.LOG_IDS:
                    LoggingSettings.LOGGER_ID = potential_log_id
                    break
                else:
                    print("❌ Please enter valid ID\n")
        
            a = input("Enter Experiment Name: ")
            a = a if a != "" else LoggingSettings.experiment_name
            LoggingSettings.experiment_name = a

            #create folder
            id_helper = LogManager()
            model_folder_path = f"Logs/{LoggingSettings.LOGGER_ID}Logs/ModelStateDicts/{id_helper.Experiment_ID}"
            os.makedirs(model_folder_path, exist_ok=True)
            PolyarithmosManager.folder_path = model_folder_path

        self._print_experiment_settings()

        input("OK? ")

        for config in configs:
            self.sleepstage = config[0]
            self.signal_type = config[1]

            print("\n" + "="*68)
            print(f"🚀 Starting experiment for {self.sleepstage} stage with {self.signal_type} signal")
            print("="*68)

            self._create_optimizer()
            self.optimizer.run_evolution()
            if LoggingSettings.LOGGING:
                id_helper = LogManager()
                experiment_id = id_helper.Experiment_ID - 1
                PolyarithmosManager.experiment_ids_within_polyartihmos.append(experiment_id)

        if LoggingSettings.LOGGING:
            polyLogger = LogManager()
            polyLogger.log_polyarithmos()

    def _generate_all_configs(self):
        configs = []

        for signal_type in Signal.ALL_SIGNALS:
            for sleep_type in Sleepstage.ALL_STAGES:
                configs.append( (sleep_type, signal_type) )

        return configs

    def _run_minimax(self):

        if LoggingSettings.LOGGING:
            while True:
                print("\n",LoggingSettings.LOG_IDS)
                potential_log_id = input("Enter logging ID: ").upper().strip()
                if potential_log_id in LoggingSettings.LOG_IDS:
                    LoggingSettings.LOGGER_ID = potential_log_id
                    break
                else:
                    print("❌ Please enter valid ID\n")
        
            a = input("Enter Experiment Name: ")
            a = a if a != "" else LoggingSettings.experiment_name
            LoggingSettings.experiment_name = a

        self._print_experiment_settings()
        self.sleepstage = Sleepstage.N3
        self.signal_type = Signal.EEG.Fpz_Cz

        input("OK? ")
        
        self._run_mini_or_max(mini=True)
        self._run_mini_or_max(mini=False)

        if LoggingSettings.LOGGING:
            poly_logger = LogManager()
            poly_logger.log_polyarithmos()

    def _run_mini_or_max(self,mini):
        if mini:
            FitnessFunctions.MINIMIZE_FITNESS = True
        else:
            FitnessFunctions.MINIMIZE_FITNESS = False

        self._create_optimizer()
        self.optimizer.run_evolution()

        if LoggingSettings.LOGGING:
            id_man = LogManager()
            PolyarithmosManager.experiment_ids_within_polyartihmos.append(id_man.Experiment_ID - 1)

    def _get_user_configuration(self):
        """Get configuration from user input"""
        skip_sleep_stage = self.args.sleep_stage is not None
        skip_signal = self.args.signal is not None
        no_logging = self.args.no_logging
        
        # Sleep stage selection
        sleep_options = [(stage, str(stage)) for stage in Sleepstage.ALL_STAGES]
        
        if skip_sleep_stage:
            self.sleepstage = self.args.sleep_stage

        else:
            print("\n📊 Available Sleep Stages:")
            for i, (stage, name) in enumerate(sleep_options, 1):
                print(f"  {i-1}. {name}")
            

            while True:
                try:
                    choice = int(input("\nSelect sleep stage (0-4): "))
                    if 0 <= choice <= 4:
                        self.sleepstage = sleep_options[choice][0]
                        break
                    print("❌ Please enter a number between 0-4")
                except ValueError:
                    print("❌ Please enter a valid number")
        
        # Signal type selection
        if EvolutionManager.SMALLER_FILES:
            print("\nWARNING: YOU ARE USING SMALLER FILES, file 'sleepEDFX/smaller_EEG_Fpz_CZ' automatically chosen")
            self.signal_type = f"smaller_{Signal.EEG.Fpz_Cz}"
        
        else:
            if skip_signal:
                self.signal_type = self.args.signal

            else:
                print("\n🔌 Available Signal Types:")
                signal_options = [
                    (Signal.EEG.Fpz_Cz, "EEG Fpz-Cz"),
                    (Signal.EEG.Pz_Oz, "EEG Pz-Oz"),
                    (Signal.EOG.HORIZONTAL, "EOG Horizontal"),
                    (Signal.EMG.SUBMENTAL, "EMG Submental")
                ]
                
                for i, (signal, name) in enumerate(signal_options, 1):
                    print(f"  {i}. {name}")
                
                while True:
                    try:
                        choice = int(input("\nSelect signal type (1-4): "))
                        if 1 <= choice <= 4:
                            self.signal_type = signal_options[choice-1][0]
                            break
                        print("❌ Please enter a number between 1-4")
                    except ValueError:
                        print("❌ Please enter a valid number")

        if no_logging:
            LoggingSettings.LOGGING = False

        else:
            print("\n📝 Logging")
            LoggingSettings.LOGGING = input("Do you want to be logging (y/*)?: ").lower().startswith('y')

            if LoggingSettings.LOGGING:
                while True:
                    print("\n",LoggingSettings.LOG_IDS)
                    potential_log_id = input("Enter logging ID: ").upper().strip()
                    if potential_log_id in LoggingSettings.LOG_IDS:
                        LoggingSettings.LOGGER_ID = potential_log_id
                        break
                    else:
                        print("❌ Please enter valid ID\n")

            else:
                LoggingSettings.LOGGER_ID = "None"
            
            if LoggingSettings.LOGGING:
                LoggingSettings.LOG_ALL_INDIVIDUALS = input("Log all individuals (y/*)?: ").lower().startswith('y')
            else:
                LoggingSettings.LOG_ALL_INDIVIDUALS = False

            if LoggingSettings.LOGGING:
                LoggingSettings.experiment_name = input("Name for Experiment: ").strip()
        
        
        self._print_experiment_settings()
        input("OK? ")

    def _create_optimizer(self):
        """Create the evolutionary optimizer with given configuration"""
        print(f"\n🔧 Creating optimizer for {self.sleepstage} stage with {self.signal_type} signal...")
        
        self.optimizer = KernelSizeEvolutionaryOptimizer(
            sleepstage=self.sleepstage,
            signal_type=self.signal_type,
        )

    def _print_experiment_settings(self):
        print("\n🧪 Experiment Configuration Summary")
        print("=" * 40)

        # Basic experiment info
        print(f"{'Sleep stage:':30} {self.sleepstage}")
        print(f"{'Signal type:':30} {self.signal_type}")
        print(f"{'Verbose:':30} {EvolutionManager.VERBOSE}")
        print(f"{'Very Verbose:':30} {EvolutionManager.VERY_VERBOSE}")

        print("\n🧬 Evolution Manager")
        print(f"{'Population size per layer:':30} {EvolutionManager.POPULATION_SIZE_PER_LAYER}")
        print(f"{'Generations:':30} {EvolutionManager.GENERATIONS}")
        print(f"{'Selection tournament size:':30} {EvolutionManager.SELECTION_TOURNAMENT_SIZE}")
        print(f"{'Elitism (per layer):':30} {EvolutionManager.ELITISM}")
        print(f"{'Hall of Fame members:':30} {EvolutionManager.HALL_OF_FAME_MEMBERS}")
        print(f"{'Max mutations:':30} {EvolutionManager.MAX_NUMBER_OF_MUTATIONS}")
        print(f"{'Crossover prob:':30} {EvolutionManager.CX_PROB}")
        print(f"{'Mutation prob:':30} {EvolutionManager.MUTATION_PROB}")
        print(f"{'Smaller files:':30} {EvolutionManager.SMALLER_FILES}")

        print("\n⏱️ Time Wall Settings")
        print(f"{'Activation generation %:':30} {TimeWall.FLIP_ON*100}%")
        print(f"{'Starting percentage:':30} {TimeWall.STARTING_PERCENTAGE*100}%")
        print(f"{'Max percentage:':30} {TimeWall.MAX_PERCENTAGE*100}%")
        print(f"{'Percentage increase:':30} {TimeWall.INCREASE*100}%")

        print("\n🏔️ Alps Manager")
        print(f"{'Aging scheme:':30} {AlpsManager.AgingScheme.uas_str}")
        print(f"{'Age gap:':30} {AlpsManager.AGE_GAP}")
        print(f"{'Max ages per layer:':30} {AlpsManager.MAX_AGE_IN_LAYERS}")
        print(f"{'Layer creation thresholds:':30} {AlpsManager.LAYER_CREATION_THRESHOLDS}")
        print(f"{'Dataset percentages:':30} {AlpsManager.percentages}")

        print("\n📦 Model Manager")
        print(f"{'Base batch size:':30} {ModelManager.BATCH_SIZE}")
        print(f"{'Learning rate:':30} {ModelManager.LEARNING_RATE}")
        print(f"{'Min kernel size:':30} {ModelManager.MIN_KERNEL_SIZE}")
        print(f"{'Max kernel size:':30} {ModelManager.MAX_KERNEL_SIZE}")
        print(f"{'Sort kernels:':30} {ModelManager.SORT_KERNELS}")
        print(f"{'Branch count range:':30} {ModelManager.NUMBER_OF_BRANCHES_RANGE}")
        print(f"{'Kernel count range:':30} {ModelManager.NUMBER_OF_KERNELS_RANGE}")

        print("\n📁 Data Manager")
        print(f"{'Dataset:':30} {DataManager.DATASET}")
        print(f"{'Available datasets:':30} {DataManager._datasets}")
        print(f"{'Max memory (MB):':30} {DataManager.MAX_MEMORY}")
        print(f"{'Even data split:':30} {DataManager.EVEN_DATA_SPLIT}")
        print(f"{'Train split:':30} {EvolutionManager.DATA_SPLIT_TRAINING}")
        print(f"{'Test split:':30} {EvolutionManager.DATA_SPLIT_TESTING}")
        print(f"{'Split valid:':30} {EvolutionManager.VALID_DATA_SPLIT}")

        print("\n📝 Logging Manager")
        print(f"{'Logging enabled:':30} {LoggingSettings.LOGGING}")
        print(f"{'Logger ID:':30} {LoggingSettings.LOGGER_ID}")
        print(f"{'Log IDs:':30} {LoggingSettings.LOG_IDS}")
        print(f"{'Log all individuals:':30} {LoggingSettings.LOG_ALL_INDIVIDUALS}")
        print(f"{'Experiment name:':30} {LoggingSettings.experiment_name}")
        print(f"{'Rounding number:':30} {LoggingTemplate.rounding_number}")
        
        print("\n💖 Fitness Manager")
        print(f"{'Fitness function:':30} {FitnessFunctions.fitness_function.__name__}")
        print(f"{'Normalization function:':30} {FitnessFunctions.normalization_function.__name__}")
        print(f"{'Minimizing Fitness:':30} {FitnessFunctions.MINIMIZE_FITNESS}")

def parse_arguments():
    """Parse command line arguments to override global settings"""
    parser = argparse.ArgumentParser(description='SLEAPy - Sleep Labeling using Evolutionary Algorithms and PyTorch')
    
    # General options
    parser.add_argument('--polyarithmos', action='store_true', help='Run all possible configurations (ultimate test mode)')
    
    # ModelManager options
    parser.add_argument('--batch-size', type=int, help=f'Batch size (default: {ModelManager.BATCH_SIZE})')
    parser.add_argument('--max-ttime', type=int, help=f'Max training time in minutes (default: {ModelManager.MAX_TIME_SPENT_TRAINING})')
    parser.add_argument('--lr', type=float, help=f'Learning rate (default: {ModelManager.LEARNING_RATE})')
    parser.add_argument('--min-ks', type=int, help=f'Minimum kernel size (default: {ModelManager.MIN_KERNEL_SIZE})')
    parser.add_argument('--max-ks', type=int, help=f'Maximum kernel size (default: {ModelManager.MAX_KERNEL_SIZE})')
    
    # EvolutionManager options
    parser.add_argument('--pop-size', type=int, help=f'Population size per layer (default: {EvolutionManager.POPULATION_SIZE_PER_LAYER})')
    parser.add_argument('--generations', type=int, help=f'Number of generations (default: {EvolutionManager.GENERATIONS})')
    parser.add_argument('--st-size', type=int, help=f'Selection tournament size (default: {EvolutionManager.SELECTION_TOURNAMENT_SIZE})')
    parser.add_argument('--hof-size', type=int, help=f'Hall of fame size (default: {EvolutionManager.HALL_OF_FAME_MEMBERS})')
    parser.add_argument('--cx-prob', type=float, help=f'Crossover probability (default: {EvolutionManager.CX_PROB})')
    parser.add_argument('--mut-prob', type=float, help=f'Mutation probability (default: {EvolutionManager.MUTATION_PROB})')
    parser.add_argument('--smaller-files', action='store_true', help=f'Use smaller files (default: {EvolutionManager.SMALLER_FILES})')
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

    return parser.parse_args()

def apply_arguments(args):
    """Apply parsed arguments to global settings"""
    # ModelManager settings
    if args.batch_size:
        ModelManager.BATCH_SIZE = args.batch_size
    if args.max_ttime:
        ModelManager.MAX_TIME_SPENT_TRAINING = args.max_training_time
    if args.lr:
        ModelManager.LEARNING_RATE = args.learning_rate
    if args.min_ks:
        ModelManager.MIN_KERNEL_SIZE = args.min_ks
    if args.max_ks:
        ModelManager.MAX_KERNEL_SIZE = args.max_ks
    
    # EvolutionManager settings
    if args.pop_size:
        EvolutionManager.POPULATION_SIZE_PER_LAYER = args.pop_size
        AlpsManager.TRAINING_SETTINGS_FOR_LAYERS = AlpsManager._get_manager(AlpsManager.percentages)
        # Update tournament size if it's based on population size
        EvolutionManager.SELECTION_TOURNAMENT_SIZE = max(3, int(EvolutionManager.POPULATION_SIZE_PER_LAYER * 0.2))
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
    if args.smaller_files:
        EvolutionManager.SMALLER_FILES = True
    if args.verbose:
        EvolutionManager.VERBOSE = True
    if args.v_verbose:
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

def main():
    """Main entry point"""
    args = parse_arguments()
    apply_arguments(args)
    
    sleapy = SLEAPy(args)

    if True:
        sleapy.run_experiment(minimax=True)


    # sleapy.run_experiment(run_every_possible_experiment=args.polyarithmos)

if __name__ == "__main__":
    try:
        sleapy_instance = main()

    except SLEAPyException as e:
        print("Exception occured during run.")
        print(e)
    except Exception as e:
        raise SLEAPyException()
