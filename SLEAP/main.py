from Globals import *
from utils import InputHandler
from utils.arg_parse import parse_arguments

from grid_search_controller.grid_search_controller import QKernel_GridSearch
from ea_controller.ea_controller import EA_Controller
from ensemble_controller.ensemble_controller import EnsembleController

class Main:
    def __init__(self):
        # self.grid_search = QKernel_GridSearch()
        self.ea_controller = EA_Controller()
        self.ensemble_controller = EnsembleController()

    def run(self):
#        print("\n\nRunning grid search for each signal and class.\n\n")
#        QKernel_GridSearch.compute_grid()

        print("\n\nRunning evolution for each signal and class.\n\n")
        self.ea_controller.run_ea()

        print("\n\nCreating ensemble with the best performing binary models.\n\n")
        self.ensemble_controller.create_ensemble()

    def __debug(self, t):
        if t == "grid":
            raise NotImplementedError(":)")
        if t == "ea":
            self.ea_controller.__single_ea(Signal.EEG.Fpz_Cz, Targets.WAKE)
        if t == "ensemble":
            self.ensemble_controller.create_ensemble()


if __name__ == "__main__":
    parse_arguments()
    main = Main()
    main.run()
