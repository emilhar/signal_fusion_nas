"""
Gives IO for SLEAP
"""

from Globals import Sleepstage, Signal, SLEAP_Exception
from EAController.KernelSizeEvolutionOptimizer import KernelSizeEvolutionaryOptimizer
from Globals import ModelManager, EvolutionManager, DataManager, LoggingManager, FitnessFunctions

class SLEAP:
    """
    Sleep
    Labeling using
    Evolutionary
    Algorithms and
    Pytorch

    Main interface for running evolutionary optimization to find optimal kernel sizes
    """

    def __init__(self):
        self.optimizer = None
        self.sleepstage = None
        self.signal_type = None
        
    def run_experiment(self, run_omega = False):
        """Run the setup and evolution process"""
        print("\n" + "="*68)
        print("🧠 SLEAP - Sleep Labeling using Evolutionary Algorithms and PyTorch")
        print("="*68)
        
        # Get user configuration
        if run_omega:
            print("\n🔥 ULTIMATE TEST MODE: Running all possible configurations")
            configs = self._generate_all_configs()

            for config in configs:
                self.sleepstage = config[0]
                self.signal_type = config[1]

                print("\n" + "="*68)
                print(f"🚀 Starting experiment for {self.sleepstage} stage with {self.signal_type} signal")
                print("="*68)

                self._create_optimizer()
                self.optimizer.run_evolution()
                self.optimizer.log_results()

        else:
            self._get_user_configuration()

            # Create optimizer with user Manager
            self._create_optimizer()
            
            # Run evolution
            self.optimizer.run_evolution()
            
            if LoggingManager.LOGGING:
                self.optimizer.log_results()

            # Show results
            self.optimizer.print_results()
    
    def _get_user_configuration(self):
        """Get configuration from user input"""
        
        # Sleep stage selection
        print("\n📊 Available Sleep Stages:")
        sleep_options = [(stage, str(stage)) for stage in Sleepstage.ALL_STAGES]
        
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

        print("\n📝 Logging")
        LoggingManager.LOGGING = input("Do you want to be logging (y/n)?: ").lower().startswith('y')

        if LoggingManager.LOGGING:
            while True:
                print("\n",LoggingManager.LOG_IDS)
                potential_log_id = input("Enter logging ID: ").upper().strip()
                if potential_log_id in LoggingManager.LOG_IDS:
                    LoggingManager.LOGGER_ID = potential_log_id
                    break
                else:
                    print("❌ Please enter valid ID\n")

        else:
            LoggingManager.LOGGER_ID = "None"
        
        if LoggingManager.LOGGING:
            LoggingManager.LOG_ALL_INDIVIDUALS = input("Log all individuals (y/n)?: ").lower().startswith('y')
        else:
            LoggingManager.LOG_ALL_INDIVIDUALS = False

        if LoggingManager.LOGGING:
            LoggingManager.experiment_name = input("Name for Experiment: ").strip()
        
        self._print_experiment_Manager()
        
        input("OK? ")
    
    def _generate_all_configs(self):
        configs = []
        
        sleep_options = Sleepstage.ALL_STAGES

        signal_options = [
            Signal.EEG.Fpz_Cz,
            Signal.EEG.Pz_Oz,
            Signal.EOG.HORIZONTAL,
            Signal.EMG.SUBMENTAL,
        ]

        for sleep_type in sleep_options:
            for signal_type in signal_options:
                configs.append( (sleep_type, signal_type) )

        return configs

    def _create_optimizer(self):
        """Create the evolutionary optimizer with given configuration"""
        print(f"\n🔧 Creating optimizer for {self.sleepstage} stage with {self.signal_type} signal...")
        
        self.optimizer = KernelSizeEvolutionaryOptimizer(
            sleepstage=self.sleepstage,
            signal_type=self.signal_type,
        )

    def _print_experiment_Manager(self):
        print("\n🧪 Experiment Configuration Summary")
        print("=" * 40)

        # Basic experiment info
        print(f"{'Sleep stage:':30} {self.sleepstage}")
        print(f"{'Signal type:':30} {self.signal_type}")
        print(f"{'Verbose:':30} {EvolutionManager.VERBOSE}")

        print("\n📦 Model Settings")
        print(f"{'Base batch size:':30} {ModelSettings.BATCH_SIZE}")
        print(f"{'Max training time:':30} {ModelSettings.MAX_TIME_SPENT_TRAINING}")
        print(f"{'Learning rate:':30} {ModelSettings.LEARNING_RATE}")
        print(f"{'Min kernel size:':30} {ModelSettings.MIN_KERNEL_SIZE}")
        print(f"{'Max kernel size:':30} {ModelSettings.MAX_KERNEL_SIZE}")
        print(f"{'Smaller files:':30} {EvolutionSettings.SMALLER_FILES}")
        print(f"{'Branch count range:':30} {ModelSettings.NUMBER_OF_BRANCHES_RANGE}")
        print(f"{'Kernel count range:':30} {ModelSettings.NUMBER_OF_KERNELS_RANGE}")

        print("\n🧬 Evolution Manager")
        print(f"{'Population size per layer:':30} {EvolutionManager.POPULATION_SIZE_PER_LAYER}")
        print(f"{'Generations:':30} {EvolutionManager.GENERATIONS}")
        print(f"{'Tournament size:':30} {EvolutionManager.SELECTION_TOURNAMENT_SIZE}")
        print(f"{'Hall of Fame members:':30} {EvolutionManager.HALL_OF_FAME_MEMBERS}")
        print(f"{'Max mutations:':30} {EvolutionManager.MAX_NUMBER_OF_MUTATIONS}")
        print(f"{'Crossover prob:':30} {EvolutionManager.CX_PROB}")
        print(f"{'Mutation prob:':30} {EvolutionManager.MUTATION_PROB}")

        print("\n📁 Data Manager")
        print(f"{'Dataset:':30} {DataManager.DATASET}")
        print(f"{'Even data split:':30} {DataManager.EVEN_DATA_SPLIT}")
        print(f"{'Train split:':30} {EvolutionManager.DATA_SPLIT_TRAINING}")
        print(f"{'Test split:':30} {EvolutionManager.DATA_SPLIT_TESTING}")
        print(f"{'Split valid:':30} {EvolutionManager.VALID_DATA_SPLIT}")

        print("\n📝 Logging Manager")
        print(f"{'Logging enabled:':30} {LoggingManager.LOGGING}")
        print(f"{'Logger ID:':30} {LoggingManager.LOGGER_ID}")
        print(f"{'Log all individuals:':30} {LoggingManager.LOG_ALL_INDIVIDUALS}")
        print(f"{'Experiment name:':30} {LoggingManager.experiment_name}")
        
        print("\n💖 Fitness Manager")
        print(f"{'Fitness function:':30} {FitnessFunctions.fitness_function.__name__}")
        print(f"{'Normalization function:':30} {FitnessFunctions.normalization_function.__name__}")
        print(f"{'Minimizing Fitness:':30} {FitnessFunctions.MINIMIZE_FITNESS}")

def main():
    """Main entry point"""
    sleap = SLEAP()
    sleap.run_experiment(run_omega=False)


if __name__ == "__main__":
    
    try:
        sleap_instance = main()
    except SLEAP_Exception as e:
        print("Exception occured during run.")
        print(e)