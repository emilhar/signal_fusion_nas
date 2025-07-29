from Globals import *
from Logs.LogManager import LogManager
from ea_controller.optimizer import KernelSizeEvolutionaryOptimizer

class ___Signal___:
    def __init__(self, n_samples):
        self.n_samples = n_samples

class EA_Controller:
    def __init__(self):
        self.batch_size = 32 # TODO: base off of how much data is present?

    def run_ea(self):
        for signal in Signal.ALL_SIGNALS:
            signal = ___Signal___(30 if signal==Signal.EMG.SUBMENTAL else 3000)
            for cls in Classes.All_CLASSES:
                self.__single_ea(signal, cls)

    def __single_ea(self, signal, cls):
        optimizer = KernelSizeEvolutionaryOptimizer(
            signal_type=signal, 
            classification_class=cls, 
            batch_size=self.batch_size,
            n_samples=signal.n_samples,
        )
        optimizer.run_evolution()

