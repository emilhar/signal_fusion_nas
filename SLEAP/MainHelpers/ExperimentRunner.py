from Globals import *
from Logs.LogManager import LogManager
from EAController.KernelSizeEvolutionOptimizer import KernelSizeEvolutionaryOptimizer
import os

def run_experiment(polyarithmos):
    """Run the setup and evolution process"""
    print("\n" + "="*68)
    print("🧠 SLEAPy - Sleep Labeling using Evolutionary Algorithms and PyTorch")
    print("="*68)

    if polyarithmos:
        _run_every_possible_experiment()
    
    else:
        optimizer = _create_optimizer()
        optimizer.run_evolution()

    
def _run_every_possible_experiment():

    print("\n🔥 ULTIMATE TEST MODE: Running all possible configurations")
    configs = _generate_all_configs()
    
    if LoggingSettings.LOGGING:
        id_helper = LogManager()
        model_folder_path = f"Logs/{LoggingSettings.LOGGER_ID}Logs/ModelStateDicts/{id_helper.Experiment_ID}"
        os.makedirs(model_folder_path, exist_ok=True)
    else:
        model_folder_path = None

    for config in configs:
        Sleepstage.current_sleepstage = config[0]
        Signal.current_signal = config[1]

        print("\n" + "="*68)
        print(f"🚀 Starting experiment for {Sleepstage.current_sleepstage} stage with {Signal.current_signal} signal")
        print("="*68)

        optimizer = _create_optimizer()
        optimizer.run_evolution(logging_folder_for_omega_runs=model_folder_path)

def _generate_all_configs():
    configs = []

    for signal_type in Signal.ALL_SIGNALS:
        for sleep_type in Sleepstage.ALL_STAGES:
            configs.append( (sleep_type, signal_type) )

    return configs

def _create_optimizer():
    """Create the evolutionary optimizer with given configuration"""
    print(f"\n🔧 Creating optimizer for {Sleepstage.current_sleepstage} stage with {Signal.current_signal} signal...")
    
    optimizer = KernelSizeEvolutionaryOptimizer(
        sleepstage=Sleepstage.current_sleepstage,
        signal_type=Signal.current_signal,
    )

    return optimizer