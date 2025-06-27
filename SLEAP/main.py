"""
Gives IO for SLEAP
"""

from Globals import Sleepstage, Signal
from EAController.KernelSizeEvolutionOptimizer import KernelSizeEvolutionaryOptimizer
from Globals import ModelSettings, EvolutionSettings, DataSettings, LoggingSettings, UniquenessFunctions

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

            # Create optimizer with user settings
            self._create_optimizer()
            
            # Run evolution
            self.optimizer.run_evolution()
            
            if LoggingSettings.LOGGING:
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
        if ModelSettings.SMALLER_FILES:
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
        LoggingSettings.LOGGING = input("Do you want to be logging (y/n)?: ").lower().startswith('y')

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
            LoggingSettings.LOG_ALL_INDIVIDUALS = input("Log all individuals (y/n)?: ").lower().startswith('y')
        else:
            LoggingSettings.LOG_ALL_INDIVIDUALS = False

        if LoggingSettings.LOGGING:
            LoggingSettings.experiment_name = input("Name for Experiment: ").strip()
                    
        self._print_experiment_settings()
        
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

    def _print_experiment_settings(self):
        print("\n🧪 Experiment Configuration Summary")
        print("=" * 40)

        # Model settings
        print(self.sleepstage)
        print(self.signal_type, '\n\n')
        print("\n📦 Model Settings")
        print(f"{'Batch size:':30} {ModelSettings.BATCH_SIZE}")
        print(f"{'Epochs per individual:':30} {ModelSettings.TRAINING_EPOCHS_PER_INDIVIDUAL}")
        print(f"{'Max training time (sec):':30} {ModelSettings.MAX_TIME_SPENT_TRAINING}")
        print(f"{'Random kernels per branch:':30} {ModelSettings.KERNELS_PER_BRANCH}")
        print(f"{'Min kernel size:':30} {ModelSettings.MIN_KERNEL_SIZE}")
        print(f"{'Max kernel size:':30} {ModelSettings.MAX_KERNEL_SIZE}")
        print(f"{'Smaller files:':30} {ModelSettings.SMALLER_FILES}")
        print(f"{'Verbose:':30} {ModelSettings.VERBOSE}")

        # Evolution settings
        print("\n🧬 Evolution Settings")
        print(f"{'Population size:':30} {EvolutionSettings.POPULATION_SIZE}")
        print(f"{'Generations:':30} {EvolutionSettings.GENERATIONS}")
        print(f"{'Tournament size:':30} {EvolutionSettings.SELECTION_TOURNAMENT_SIZE}")
        print(f"{'Crossover probability:':30} {EvolutionSettings.CX_PROB}")
        print(f"{'Mutation probability:':30} {EvolutionSettings.MUTATION_PROB}")
        print(f"{'Data points per individual:':30} {EvolutionSettings.DATA_POINTS_PER_INDIVIUAL}")
        print(f"{'Data for training:':30} {EvolutionSettings.DATA_SPLIT_TRAINING}")
        print(f"{'Data for testing:':30} {EvolutionSettings.DATA_SPLIT_TESTING}")

        # Dataset info
        print("\n📁 Data Settings")
        print(f"{'Dataset:':30} {DataSettings.DATASET}")

        # Logging settings
        print("\n📝 Logging Settings")
        print(f"{'Logging:':30} {LoggingSettings.LOGGING}")
        print(f"{'Logging ID:':30} {LoggingSettings.LOGGER_ID}")
        print(f"{'Log all individuals:':30} {LoggingSettings.LOG_ALL_INDIVIDUALS}")
        
        print("\n🦠 Uniqueness Settings")
        print(f"{'Uniqueness function:':30} {UniquenessFunctions.uniqueness_function.__name__}")

        print("\n" + "=" * 40 + "\n")



def main():
    """Main entry point"""
    sleap = SLEAP()
    sleap.run_experiment(run_omega=False)


if __name__ == "__main__":
    sleap_instance = main()