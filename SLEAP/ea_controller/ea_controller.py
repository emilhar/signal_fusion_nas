from Globals import *
from ea_controller.optimizer import KernelSizeEvolutionaryOptimizer

class ___Signal___:
    def __init__(self, n_samples):
        self.n_samples = n_samples

class EA_Controller:
    def __init__(self):
        self.batch_size = 32 # TODO: base off of how much data is present?

    def run_ea(self, targets, signals):
        
        for signal in signals:
            for cls in targets:
                self.__single_ea(signal, cls, part_of_bigger_ea=True)

    def __single_ea(self, signal, cls, part_of_bigger_ea=False):
        optimizer = KernelSizeEvolutionaryOptimizer(
            signal_type=signal.name,
            n_samples=signal.n_samples,
            classification_class=cls,
            batch_size=self.batch_size,
        )
        optimizer.run_evolution(part_of_bigger_ea)

