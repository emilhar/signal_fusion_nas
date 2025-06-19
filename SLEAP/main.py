"""
Gives IO for SLEAP
"""

from Globals import Sleepstage, Signal
from EAController.KernelSizeEvolutionOptimizer import KernelSizeEvolutionaryOptimizer
from Globals import ModelSettings, EvolutionSettings, DataSettings, LoggingSettings

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
        
    def run_experiment(self, run_all_experiment_configs = False):
        """Run the setup and evolution process"""
        print("\n" + "="*68)
        print("🧠 SLEAP - Sleep Labeling using Evolutionary Algorithms and PyTorch")
        print("="*68)
        

        # Get user configuration
        if run_all_experiment_configs:
            print("\n🔥 ULTIMATE TEST MODE: Running all possible configurations")
            # configs = self._generate_all_configs()

            # for config in configs:

            #     print("\n" + "="*68)
            #     print(f"🚀 Starting experiment for {config['sleepstage']} stage with {config['signal_type']} signal")
            #     print("="*68)

            #     self._create_optimizer(config)
            #     self.optimizer.run_evolution()
            #     self.optimizer.log_results()


        else:
            config = self._get_user_configuration()
        
            # Create optimizer with user settings
            self._create_optimizer(config)
            
            # Run evolution
            self.optimizer.run_evolution()
            
            if EvolutionSettings.LOGGING:
                self.optimizer.log_results()

            # Show results
            self.optimizer.print_results()
    
    def _get_user_configuration(self):
        """Get configuration from user input"""
        config = {}
        
        # Sleep stage selection
        print("\n📊 Available Sleep Stages:")
        sleep_options = [
            (Sleepstage.WAKE, "Wake"),
            (Sleepstage.LIGHT_SLEEP, "Light Sleep"),
            (Sleepstage.DEEP_SLEEP, "Deep Sleep"),
            (Sleepstage.REM, "REM Sleep")
        ]
        
        for i, (stage, name) in enumerate(sleep_options, 1):
            print(f"  {i}. {name}")
        
        while True:
            try:
                choice = int(input("\nSelect sleep stage (1-4): "))
                if 1 <= choice <= 4:
                    config['sleepstage'] = sleep_options[choice-1][0]
                    break
                print("❌ Please enter a number between 1-4")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Signal type selection
        if ModelSettings.SMALLER_FILES:
            print("\nWARNING: YOU ARE USING SMALLER FILES, file 'smaller_EEG_Fpz_CZ' automatically chosen")
            config['signal_type'] = f"smaller_{Signal.EEG.Fpz_Cz}"
        
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
                        config['signal_type'] = signal_options[choice-1][0]
                        break
                    print("❌ Please enter a number between 1-4")
                except ValueError:
                    print("❌ Please enter a valid number")
        
        self._print_experiment_settings()
        
        input("OK? ")

        return config
    
    def _create_optimizer(self, config):
        """Create the evolutionary optimizer with given configuration"""
        print(f"\n🔧 Creating optimizer for {config['sleepstage']} stage with {config['signal_type']} signal...")
        
        self.sleepstage = config['sleepstage']
        self.signal_type = config['signal_type']
        
        self.optimizer = KernelSizeEvolutionaryOptimizer(
            sleepstage=config['sleepstage'],
            signal_type=config['signal_type'],
        )

    def _print_experiment_settings(self):
        print("\n🧪 Experiment Configuration Summary")
        print("=" * 40)

        # Model settings
        print("\n📦 Model Settings")
        print(f"{'Batch size:':30} {ModelSettings.BATCH_SIZE}")
        print(f"{'Epochs per individual:':30} {ModelSettings.TRAINING_EPOCHS_PER_INDIVIDUAL}")
        print(f"{'Dataset fraction:':30} {ModelSettings.DATASET_FRACTION}")
        print(f"{'Max training time (sec):':30} {ModelSettings.MAX_TIME_SPENT_TRAINING}")
        print(f"{'Kernel sizes:':30} {ModelSettings.KERNEL}")
        print(f"{'Min kernel size:':30} {ModelSettings.MIN_KERNEL_SIZE}")
        print(f"{'Max kernel size:':30} {ModelSettings.MAX_KERNEL_SIZE}")
        print(f"{'Smaller files:':30} {ModelSettings.SMALLER_FILES}")
        print(f"{'Verbose:':30} {ModelSettings.VERBOSE}")

        # Evolution settings
        print("\n🧬 Evolution Settings")
        print(f"{'Population size:':30} {EvolutionSettings.POPULATION_SIZE}")
        print(f"{'Generations:':30} {EvolutionSettings.GENERATIONS}")
        print(f"{'Tournament size:':30} {EvolutionSettings.TOURNAMENT_SIZE}")
        print(f"{'Crossover probability:':30} {EvolutionSettings.CX_PROB}")
        print(f"{'Mutation probability:':30} {EvolutionSettings.MUTATION_PROB}")
        print(f"{'Offspring variation:':30} {EvolutionSettings.OFFSPRING_VARIATION}")
        print(f"{'Layers of CNN:':30} {EvolutionSettings.LAYERS_OF_CNN}")
        print(f"{'Random kernels per branch:':30} {EvolutionSettings.RANDOM_KERNELS_PER_BRANCH}")
        print(f"{'Data points per individual:':30} {EvolutionSettings.DATA_POINTS_PER_INDIVIUAL}")

        # Tournament of Champions
        print("\n🏆 Tournament of Champions")
        print(f"{'Enabled:':30} {EvolutionSettings.TOC_ON}")
        print(f"{'Generations between:':30} {EvolutionSettings.TOC_GENERATIONS_BETWEEN}")
        print(f"{'Tournament size:':30} {EvolutionSettings.TOC_TOURNAMENT_SIZE}")
        print(f"{'TOC batch size:':30} {EvolutionSettings.TOC_BATCH_SIZE}")

        # Dataset info
        print("\n📁 Data Settings")
        print(f"{'Dataset:':30} {DataSettings.DATASET}")

        # Logging settings
        print("\n📝 Logging Settings")
        print(f"{'Log all individuals:':30} {LoggingSettings.LOG_INDIVIDUALS}")

        print("\n" + "=" * 40 + "\n")



def main():
    """Main entry point"""
    sleap = SLEAP()
    sleap.run_experiment()


if __name__ == "__main__":
    sleap_instance = main()