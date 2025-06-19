"""
Gives IO for SLEAP
"""

from Globals import Sleepstage, Signal
from EAController.KernelSizeEvolutionOptimizer import KernelSizeEvolutionaryOptimizer
from Globals import ModelSettings, EvolutionSettings

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
        
    def run_interactive(self):
        """Run the interactive setup and evolution process"""
        print("\n" + "="*68)
        print("🧠 SLEAP - Sleep Labeling using Evolutionary Algorithms and PyTorch")
        print("="*68)
        
        # Get user configuration
        config = self._get_user_configuration()
        
        # Create optimizer with user settings
        self._create_optimizer(config)
        
        # Run evolution
        self._run_evolution()
        
        if EvolutionSettings.LOGGING:
            self.optimizer.log_results()

        # Show results
        self.optimizer.print_results()
        
        return self.optimizer
    
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
        
        # Training parameters
        print("\n🎯 Training Configuration:")
        config['batch_size'] = ModelSettings.BATCH_SIZE
        config['epochs_per_individual'] = ModelSettings.TRAINING_EPOCHS_PER_INDIVIDUAL
        config['dataset_fraction'] = ModelSettings.DATASET_FRACTION

        print(f"{'Batch Size:':25} {config['batch_size']}")
        print(f"{'Epochs Per Individual:':25} {config['epochs_per_individual']}")
        print(f"{'Dataset Fraction:':25} {config['dataset_fraction']}")

        # Evolution parameters
        print("\n🧬 Evolution Configuration:")
        config['population_size'] = EvolutionSettings.POPULATION_SIZE
        config['generations'] = EvolutionSettings.GENERATIONS
        config['cx_prob'] = EvolutionSettings.CX_PROB
        config['mut_prob'] = EvolutionSettings.MUTATION_PROB

        print(f"{'Population Size:':25} {config['population_size']}")
        print(f"{'Generations:':25} {config['generations']}")
        print(f"{'Crossover Probability:':25} {config['cx_prob']}")
        print(f"{'Mutation Probability:':25} {config['mut_prob']}")

        # Verbosity
        config['verbose'] = ModelSettings.VERBOSE
        print(f"\n{'🔊 Verbose Output:':25} {ModelSettings.VERBOSE}")

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
            batch_size=config.get('batch_size', ModelSettings.BATCH_SIZE),
            epochs_per_individual=config.get('epochs_per_individual', ModelSettings.TRAINING_EPOCHS_PER_INDIVIDUAL),
            dataset_fraction=config.get('dataset_fraction', ModelSettings.DATASET_FRACTION),
            population_size=config.get('population_size', ModelSettings.DATASET_FRACTION),
            generations=config.get('generations', EvolutionSettings.GENERATIONS),
            cx_prob=config.get('cx_prob', EvolutionSettings.CX_PROB),
            mut_prob=config.get('mut_prob', EvolutionSettings.MUTATION_PROB),
            verbose=config.get('verbose', ModelSettings.VERBOSE),
        )
    
    def _run_evolution(self):
        """Run the evolutionary algorithm"""
        print("\n🚀 Starting evolutionary optimization...")
        print("This may take a while depending on your configuration...\n")
        

        population, hall_of_fame, stats = self.optimizer.run_evolution()
        print("\n✅ Evolution completed successfully!")
        return population, hall_of_fame, stats




def main():
    """Main entry point"""
    sleap = SLEAP()
    
    sleap.run_interactive()
    
    return sleap


if __name__ == "__main__":
    sleap_instance = main()