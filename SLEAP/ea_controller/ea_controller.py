from Globals import *
from Logs.LogManager import LogManager
from ea_controller.optimizer import KernelSizeEvolutionaryOptimizer

class EA_Controller:
    def __init__(self):
        self.batch_size = 32 # TODO: base off of how much data is present?

    def run_ea(self):
        for signal in Signal.ALL_SIGNALS:
            for cls in Classes.All_CLASSES:
                self.__single_ea(signal, cls, part_of_bigger_ea=True)

    def __single_ea(self, signal, cls, part_of_bigger_ea=False):
        optimizer = KernelSizeEvolutionaryOptimizer(
            signal_type=signal, 
            classification_class=cls, 
            batch_size=self.batch_size
        )
        optimizer.run_evolution(part_of_bigger_ea)

