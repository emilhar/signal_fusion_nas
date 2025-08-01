from Globals import *
from datahelpers.datahelper import prepare_data

#from grid_search_controller.grid_search_controller import QKernel_GridSearch
from ea_controller.ea_controller import EA_Controller
from ensemble_controller.ensemble_controller import EnsembleController

from datahelpers.data import Data
from utils.clr import Clr

class Main:
    def __init__(self):
        
        self.targets, self.signals = prepare_data(mb_per_part=0.5)
        # self.grid_search = QKernel_GridSearch()
        self.ea_controller = EA_Controller()
        self.ensemble_controller = EnsembleController(self.targets, self.signals, debug=True)

    def run(self):
        while True:
            target_ranking = self.ensemble_controller.create_ensemble(use_temp=False)
            for target in target_ranking:
                # self.ea_controller.run_ea(target)
                new_target_ranking = self.ensemble_controller.create_ensemble(use_temp=True)
                for original, new in zip(target_ranking, new_target_ranking):
                    original_name, original_score = original
                    new_name, new_score = new
                    
                    arrow_color = "green" if new_score >= original_score else "red"
                    colored_arrow = Clr("--->", arrow_color)
                    
                    print(
                        f"Original: {original_name}: {original_score:.2f} "
                        f"{colored_arrow} "
                        f"New: {new_name}: {new_score:.2f}\n\n"
                    )
                break

                # if better, replace AND others have not suffered
                    # break
                    # save new target to saved_models
            break


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
    main.run()
