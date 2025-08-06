from Globals import *
from datahelpers.datahelper import prepare_data

#from grid_search_controller.grid_search_controller import QKernel_GridSearch
from ea_controller.ea_controller import EA_Controller
from ensemble_controller.ensemble_controller import EnsembleController
from utils.trained_model_maker import TrainedModelMaker

from datahelpers.data import Data
from datahelpers.target import Target
from datahelpers.signal import Signal
from utils.clr import Clr
import os
import shutil

from logger import Logger

class Main:
    def __init__(self):
        self.targets, self.signals = prepare_data(mb_per_part=0.5)
        self.ea_controller = EA_Controller()
        self.ensemble_controller = EnsembleController(self.targets, self.signals)
        self.max_filters = 4
        self.clear_temp_models()

    def run(self):
        Logger.log_new_experiment_heading()
        self.ensemble_controller.get_initial_models()
        while True:
            target: Target
            for target in self.targets:
                target_ranking = self.ensemble_controller.create_ensemble(pos_idx=target.data_label, use_temp=False)
                Logger.log_ensemble(target_ranking, fake=False)
                self.clear_temp_models()
                Logger.log_ea_start(target)
                for (logbook, signal, t) in self.ea_controller.run_ea(target):
                    Logger.log_ea_logbook(logbook, signal, t)

                new_target_ranking = self.ensemble_controller.create_ensemble(pos_idx=target.data_label, use_temp=True)
                Logger.log_ensemble(new_target_ranking, fake=True)

                target_change = 0
                other_change = 0

                for original, new in zip(target_ranking, new_target_ranking):
                    original, original_score = original
                    new, new_score = new
                    
                    arrow_color = "green" if new_score >= original_score else "red"
                    colored_arrow = Clr("--->", arrow_color)

                    print_str = (
                        f"Original: {original}: {original_score:.2f} "
                        f"{colored_arrow} "
                        f"New: {new}: {new_score:.2f}\n"
                    )
                    print(print_str)
                    if original.given_name == target.given_name:
                        target_change = (new_score - original_score) / original_score
                    else:
                        other_change += (new_score - original_score) / original_score

                self.log_ranking_comparison(target_ranking, new_target_ranking, use_table=True)
                if target_change > 0.0:
                    print_str = f"TARGET {target} IMPROVED"
                    print(print_str)
                    Logger.log_line(print_str)
                    Logger.log_successful_upgrade()
                    self.move_from_temp_to_saved()
                else:
                    print_str = f"TARGET {target} NOT IMPROVED, MOVING TO NEXT"
                    print(print_str)
                    Logger.log_line(print_str)

            print_str = "ADDING MORE FILTERS..."
            print(print_str)
            Logger.log_line(print_str)
            old_filter_count = TrainedModelMaker.NUM_FILTERS
            progress = self.update_filters()
            Logger.log_failed_upgrade(old_filter_count, new_filter_count=TrainedModelMaker.NUM_FILTERS)

            # if no meaningful progress, terminate main loop.
            if not progress:
                print_str = "NO PROGRESS MADE, TERMINATING."
                print(print_str)
                Logger.log_line(print_str)
                Logger.log_completion(target_ranking)
                break

    def move_from_temp_to_saved(self):
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

        self.clear_temp_models()

    def clear_temp_models(self):
        if os.path.exists("temp_models"):
            shutil.rmtree("temp_models")
            os.makedirs("temp_models")
        else:
            raise ValueError("TEMP MODELS???")

    def update_filters(self):
        if TrainedModelMaker.NUM_FILTERS * 2 > self.max_filters:
            print_str = "REACHED MAX FILTERS PER CONVOLUTION"
            print(print_str)
            Logger.log_line(print_str)
            return False
        
        TrainedModelMaker.NUM_FILTERS *= 2

        self.ensemble_controller.update_filters_for_binary_models()
        return True
    

    def log_ranking_comparison(self, target_ranking, new_target_ranking, use_table=False):
        if not target_ranking or not new_target_ranking:
            Logger.log_line("> No ranking data to compare\n")
            return
        
        # Header
        Logger.log_line("\n## Ranking Comparison\n", use_timestamp=False)
        
        if use_table:
            # Table version
            Logger.log_line("| Original | Score | → | New | Score | Change |", use_timestamp=False)
            Logger.log_line("|----------|-------|---|-----|-------|--------|", use_timestamp=False)
            
            for (original, original_score), (new, new_score) in zip(target_ranking, new_target_ranking):
                color = "green" if new_score >= original_score else "red"
                
                Logger.log_line(
                    f"| `{original}` | `{original_score:.2f}` | → | `{new}` | `{new_score:.2f}` | "
                    f"<span style='color:{color}'>{"▅"}</span> |",
                    use_timestamp=False
                )
        else:
            # List version
            Logger.log_line("### Changes:\n")
            
            for (original, original_score), (new, new_score) in zip(target_ranking, new_target_ranking):
                direction = "↑" if new_score >= original_score else "↓"
                color = "green" if new_score >= original_score else "red"
                colored_arrow = Clr("→", color)
                
                Logger.log_line(
                    f"- `{original}`: `{original_score:.2f}` {colored_arrow} "
                    f"`{new}`: `{new_score:.2f}` {direction}",
                    use_timestamp=False
                )
        
    


if __name__ == "__main__":
    #parse_arguments()
    main = Main()
    main.run()