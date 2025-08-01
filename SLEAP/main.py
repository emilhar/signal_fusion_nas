import gc

from Globals import *
from datahelpers.datahelper import prepare_data

#from grid_search_controller.grid_search_controller import QKernel_GridSearch
from ea_controller.ea_controller import EA_Controller
from ensemble_controller.ensemble_controller import EnsembleController
from utils.trained_model_maker import TrainedModelMaker

from datahelpers.data import Data
from utils.clr import Clr

class Main:
    def __init__(self):
        
        self.targets, self.signals = prepare_data(mb_per_part=0.5)
        # self.grid_search = QKernel_GridSearch()
        self.ea_controller = EA_Controller()
        self.ensemble_controller = EnsembleController(self.targets, self.signals, debug=True)
        self.max_filters = 2

    def run(self):
        patience = 2
        target_run_count = {target.given_name: 0 for target in self.targets}
        #self.ensemble_controller.get_initial_models()
        while True:
            progress = False
            target_ranking = self.ensemble_controller.create_ensemble(use_temp=False)
            gc.collect()
    
            for x in target_ranking:
                target, score = x
                if target_run_count[target.given_name] >= patience:
                    continue

                self.ea_controller.run_ea(target)
                new_target_ranking = self.ensemble_controller.create_ensemble(use_temp=True)

                target_change = 0
                other_change = 0
                for original, new in zip(target_ranking, new_target_ranking):
                    original, original_score = original
                    new, new_score = new
                    
                    arrow_color = "green" if new_score >= original_score else "red"
                    colored_arrow = Clr("--->", arrow_color)
                    
                    print(
                        f"Original: {original}: {original_score:.2f} "
                        f"{colored_arrow} "
                        f"New: {new}: {new_score:.2f}\n"
                    )
                    if original == target:
                        target_change = (new_score - original_score) / original_score
                    else:
                        other_change += (new_score - original_score) / original_score

                if target_change >= 1 and other_change >= -1:
                    print("TARGET IMPROVED")
                    progress = True
                    break
                
                print("TARGET NOT IMPROVED, MOVING TO NEXT")
                target_run_count[target] += 1

            else:
                print("UPDATING FILTERS...")
                progress = self.update_filters()
                

            # if no meaningful progress, terminate main loop.
            if not progress:
                print("NO PROGRESS MADE, TERMINATING")
                break


    def update_filters(self):
        if TrainedModelMaker.NUM_FILTERS * 2 > self.max_filters:
            print("REACHED MAX FILTERS PER CONVOLUTION")
            return False
        
        TrainedModelMaker.NUM_FILTERS *= 2

        self.ensemble_controller.update_filters_for_binary_models()
        quit()
        return True
    

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
