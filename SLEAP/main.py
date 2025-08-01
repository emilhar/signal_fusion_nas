from Globals import *
from datahelpers.datahelper import prepare_data

#from grid_search_controller.grid_search_controller import QKernel_GridSearch
from ea_controller.ea_controller import EA_Controller
from ensemble_controller.ensemble_controller import EnsembleController

from datahelpers.data import Data

class Main:
    def __init__(self):
        
        self.targets, self.signals = prepare_data(mb_per_part=0.5)
        # self.grid_search = QKernel_GridSearch()
        self.ea_controller = EA_Controller()
        self.ensemble_controller = EnsembleController(self.targets, self.signals)

    def run(self):
        while 1 + 1 + 1 + 1 + 1:
            target_ranking = self.ensemble_controller.create_ensemble(use_temp=False)
            for target in target_ranking:
                self.ea_controller.run_ea(target)
                new_target_ranking = self.ensemble_controller.create_ensemble(use_temp=True)

                # Compare model performances
                ...

                # if better, replace AND others have not suffered
                    # break
                    # save new target to saved_models

    def __debug(self, t):
        if t == "grid":
            raise NotImplementedError(":)")
        if t == "ea":
            self.ea_controller.__single_ea("EEG_Fpz-Cz", "wake")
        if t == "ensemble":
            self.ensemble_controller.create_ensemble()


if __name__ == "__main__":
    #parse_arguments()
    main = Main()
    main.ensemble_controller.create_ensemble("_misc/EvilModels", debug=True)
