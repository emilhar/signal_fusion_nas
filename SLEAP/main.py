from Globals import *
from datahelpers.datahelper import prepare_data

#from grid_search_controller.grid_search_controller import QKernel_GridSearch
from ea_controller.ea_controller import EA_Controller
from ensemble_controller.ensemble_controller import EnsembleController
from utils.trained_model_maker import TrainedModelMaker

from datahelpers.data import Data
from utils.clr import Clr
import os
import shutil

from logger import Logger

class Main:
    def __init__(self):
        self.targets, self.signals = prepare_data(mb_per_part=0.5)
        self.ea_controller = EA_Controller()
        self.ensemble_controller = EnsembleController(self.targets, self.signals, debug=True)
        self.max_filters = 2
        if os.path.exists("temp_models"):
            shutil.rmtree("temp_models")
            os.makedirs("temp_models")

    def run(self):
        patience = 2
        target_run_count = {target.given_name: 0 for target in self.targets}
        Logger.log_new_experiment_heading()
        #self.ensemble_controller.get_initial_models()
        while True:
            progress = False
            target_ranking = self.ensemble_controller.create_ensemble(use_temp=False)
            Logger.log_ensemble(target_ranking, fake=False)
    
            for x in target_ranking:
                target, score = x
                if target_run_count[target.given_name] >= patience:
                    continue

                Logger.log_ea_start(target)
                self.ea_controller.run_ea(target)

                new_target_ranking = self.ensemble_controller.create_ensemble(use_temp=True)
                Logger.log_ensemble(new_target_ranking, fake=True)

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
                    print(f"TARGET {target} IMPROVED")
                    Logger.log_successful_upgrade()
                    self.move_from_temp_to_saved()
                    progress = True
                    break
                
                print(f"TARGET {target} NOT IMPROVED, MOVING TO NEXT")
                target_run_count[target.given_name] += 1

            else:
                print("ADDING MORE FILTERS...")
                old_filter_count = TrainedModelMaker.NUM_FILTERS
                progress = self.update_filters()
                Logger.log_failed_upgrade(old_filter_count, new_filter_count=TrainedModelMaker.NUM_FILTERS)

            # if no meaningful progress, terminate main loop.
            if not progress:
                print("NO PROGRESS MADE, TERMINATING.")
                Logger.log_completion(target_ranking)
                break

    def move_from_temp_to_saved():
        """
        Replace everything in the "saved_models" folder with files of the same name
        from the "temp_models" folder.
        """

        # Define the folder paths
        temp_folder = "temp_models"
        saved_folder = "saved_models"

        # Iterate through files in temp_models
        for filename in os.listdir(temp_folder):
            temp_path = os.path.join(temp_folder, filename)
            saved_path = os.path.join(saved_folder, filename)
            
            # If it's a file (not a directory), copy it to saved_models
            if os.path.isfile(temp_path):
                shutil.copy2(temp_path, saved_path)

        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        os.makedirs(temp_folder)

    def update_filters(self):
        if TrainedModelMaker.NUM_FILTERS * 2 > self.max_filters:
            print("REACHED MAX FILTERS PER CONVOLUTION")
            return False
        
        TrainedModelMaker.NUM_FILTERS *= 2

        self.ensemble_controller.update_filters_for_binary_models()
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
