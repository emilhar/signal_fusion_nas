from Globals import *
from ea_controller.optimizer import KernelSizeEvolutionaryOptimizer
from datahelpers.data import Data
import shutil
import os

class ___Signal___:
    def __init__(self, n_samples):
        self.n_samples = n_samples

class EA_Controller:
    def __init__(self):
        self.batch_size = 32 # TODO: base off of how much data is present?

    def run_ea(self, target_to_update):
        
        if target_to_update.given_name not in Data.get_all_target_names():
            raise ValueError(f"Target does not exist: {target_to_update}")

        temp_file_path = "temp_models"
        if os.path.exists(temp_file_path):
            shutil.rmtree(temp_file_path)
        os.makedirs(temp_file_path)

        da = Data()
        for signal in da.signal_objects:
            self.__single_ea(signal, target_to_update, part_of_bigger_ea=True)

    def __single_ea(self, signal, cls, part_of_bigger_ea=False):
        optimizer = KernelSizeEvolutionaryOptimizer(
            signal_type=signal.name,
            n_samples=signal.n_samples,
            classification_class=cls,
            batch_size=self.batch_size,
        )
        optimizer.run_evolution(part_of_bigger_ea)  # Side effect: saves the best model to temp_models
