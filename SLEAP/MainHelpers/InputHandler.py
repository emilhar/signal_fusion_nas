from Globals import *

def get_user_configuration(args):
    """Get configuration from user input"""
    skip_sleep_stage = args.sleep_stage is not None
    skip_signal = args.signal is not None
    no_logging = args.no_logging
    
    if skip_sleep_stage:
        Sleepstage.current_sleepstage = args.sleep_stage

    else:
        print("\n📊 Available Sleep Stages:")

        sleep_options = [(stage, str(stage)) for stage in Sleepstage.ALL_STAGES]
        for i, (_, name) in enumerate(sleep_options):
            print(f"  {i}. {name}")
        
        while True:
            try:
                choice = int(input(f"\nSelect sleep stage (0-{len(sleep_options)-1}): "))
                if 0 <= choice <= len(sleep_options)-1:
                    Sleepstage.current_sleepstage = sleep_options[choice][0]
                    break
                print(f"❌ Please enter a number between 0 - {len(sleep_options)-1}")
            except ValueError:
                print("❌ Please enter a valid number")

    if skip_signal:
        Signal.current_signal = args.signal

    else:
        print("\n🔌 Available Signal Types:")

        signal_options = [(signal, str(signal)) for signal in Signal.ALL_SIGNALS]
        for i, (_, name) in enumerate(signal_options):
            print(f"  {i}. {name}")
        
        while True:
            try:
                choice = int(input(f"\nSelect signal types (0-{len(signal_options)-1}): "))
                if 0 <= choice <= len(signal_options)-1:
                    Signal.current_signal = signal_options[choice][0]
                    break
                print(f"❌ Please enter a number between 0 - {len(signal_options)-1}")
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

            LoggingSettings.LOG_ALL_INDIVIDUALS = input("Log all individuals (y/*)?: ").lower().startswith('y')
            LoggingSettings.experiment_name = input("Name for Experiment: ").strip()

        else:
            LoggingSettings.LOGGER_ID = "None"
            LoggingSettings.LOG_ALL_INDIVIDUALS = False
            LoggingSettings.experiment_name = "None"
    
    _print_experiment_settings(user_check=True)


def _print_experiment_settings(user_check = True):
    print("\n🧪 Experiment Configuration Summary")
    print("=" * 40)

    # Basic experiment info
    print(f"{'Sleep stage:':30} {Sleepstage.current_sleepstage}")
    print(f"{'Signal type:':30} {Signal.current_signal}")
    print(f"{'Verbose:':30} {EvolutionManager.VERBOSE}")
    print(f"{'Very Verbose:':30} {EvolutionManager.VERY_VERBOSE}")

    print("\n🧬 Evolution Manager")
    print(f"{'Population size:':30} {EvolutionManager.POPULATION_SIZE}")
    print(f"{'Generations:':30} {EvolutionManager.GENERATIONS}")
    print(f"{'Selection tournament size:':30} {EvolutionManager.SELECTION_TOURNAMENT_SIZE}")
    print(f"{'Elitism:':30} {EvolutionManager.ELITISM}")
    print(f"{'Hall of Fame members:':30} {EvolutionManager.HALL_OF_FAME_MEMBERS}")
    print(f"{'Max mutations:':30} {EvolutionManager.MAX_NUMBER_OF_MUTATIONS}")
    print(f"{'Crossover prob:':30} {EvolutionManager.CX_PROB}")
    print(f"{'Mutation prob:':30} {EvolutionManager.MUTATION_PROB}")

    print("\n📦 Model Manager")
    print(f"{'Base batch size:':30} {ModelManager.BATCH_SIZE}")
    print(f"{'Learning rate:':30} {ModelManager.LEARNING_RATE}")
    print(f"{'Min kernel size:':30} {ModelManager.MIN_KERNEL_SIZE}")
    print(f"{'Max kernel size:':30} {ModelManager.MAX_KERNEL_SIZE}")
    print(f"{'Branch count range:':30} {ModelManager.NUMBER_OF_BRANCHES_RANGE}")
    print(f"{'Kernel count range:':30} {ModelManager.NUMBER_OF_KERNELS_RANGE}")

    print("\n📁 Data Manager")
    print(f"{'Dataset:':30} {DataManager.DATASET}")
    print(f"{'Available datasets:':30} {DataManager._datasets}")
    print(f"{'Max memory (MB):':30} {DataManager.MAX_MEMORY}")
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

    if user_check:
        input("OK? ")