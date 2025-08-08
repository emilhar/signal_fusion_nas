from Globals import *
from datahelpers.datahelper import prepare_data

#from grid_search_controller.grid_search_controller import QKernel_GridSearch
from ea_controller.ea_controller import EA_Controller
from ensemble_controller.ensemble_controller import EnsembleController
from utils.trained_model_maker import TrainedModelMaker

from utils.arg_parse import parse_arguments

from datahelpers.target import Target
from utils.clr import Clr
import os
import shutil

from logger import Logger

class Main:
    def __init__(self):
        self.targets, self.signals = prepare_data(mb_per_part=128)
        self.ea_controller = EA_Controller()
        self.ensemble_controller = EnsembleController(self.targets, self.signals)
        #self.clear_saved_models()
        #self.clear_temp_models()

    def run(self):
        avg_target_ranking = self._run_initial_ensemble_evaluation(num_evals=1)
        Logger.log_ensemble(avg_target_ranking, fake=False)
        
        # self._run_evolutionary_algorithm()
        
        # avg_new_target_ranking = self._run_post_ea_ensemble_evaluation(num_evals=5)
        # Logger.log_ensemble(avg_new_target_ranking, fake=True)
        
        # change = self._compare_rankings(avg_target_ranking, avg_new_target_ranking)
        # self._log_results(change)

    def _run_initial_ensemble_evaluation(self, num_evals):
        """Runs with filters = max_filters"""
        Logger.log_new_experiment_heading()
        self.ensemble_controller.get_initial_models()
        
        target_rankings = []
        for _ in range(num_evals):
            ranking = self.ensemble_controller.create_ensemble(use_temp=False)
            target_rankings.append(ranking)
        
        return self._calculate_average_ranking(target_rankings)

    def _run_evolutionary_algorithm(self):
        """Runs with filters = 1"""
        TrainedModelMaker.NUM_FILTERS = 1
        for target in self.targets:
            Logger.log_ea_start(target)
            for (logbook, signal, t) in self.ea_controller.run_ea(target):
                Logger.log_ea_logbook(logbook, signal, t)

    def _run_post_ea_ensemble_evaluation(self, num_evals):
        """Runs with filters = max_filters"""
        new_target_rankings = []
        for _ in range(num_evals):
            new_ranking = self.ensemble_controller.create_ensemble(use_temp=True)
            new_target_rankings.append(new_ranking)
        return self._calculate_average_ranking(new_target_rankings)

    def _compare_rankings(self, original_ranking, new_ranking):
        change = 0
        for original, new in zip(original_ranking, new_ranking):
            original, original_score = original
            new, new_score = new
            
            arrow_color = "green" if new_score >= original_score else "red"
            colored_arrow = Clr("--->", arrow_color)

            print_str = (
                f"Original: {original}: {original_score:.2f} "
                f"{colored_arrow} "
                f"New: {new}: {new_score:.2f}\n"
            )
            change += (new_score - original_score) / original_score
            print(print_str)

        Logger.log_ranking_comparison(original_ranking, new_ranking, use_table=True)
        return change

    def _log_results(self, change):
        if change > 0.0:
            print_str = f"ENSEMBLE IMPROVED AFTER EAs"
            print(print_str)
            Logger.log_line(print_str)
            Logger.log_successful_upgrade()
            self.move_from_temp_to_saved()
        else:
            print_str = f"ENSEMBLE NOT IMPROVED AFTER EAs"
            print(print_str)
            Logger.log_line(print_str)


    def _calculate_average_ranking(self, rankings_list):
        sums = {}
        counts = {}
        for ranking in rankings_list:
            for target, value in ranking:
                if target not in sums:
                    sums[target] = 0.0
                    counts[target] = 0
                sums[target] += value
                counts[target] += 1
        
        return [(target, sums[target]/counts[target]) for target in sums]

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

    def clear_saved_models(self):
        if os.path.exists("saved_models"):
            shutil.rmtree("saved_models")
            os.makedirs("saved_models")
        else:
            raise ValueError("SAVED MODEEELS???")

    def clear_temp_models(self):
        if os.path.exists("temp_models"):
            shutil.rmtree("temp_models")
            os.makedirs("temp_models")
        else:
            raise ValueError("TEMP MODELS???")

    def update_filters(self):
        if TrainedModelMaker.NUM_FILTERS * 2 > self.max_filters:
            print_str = "## REACHED MAX FILTERS PER CONVOLUTION"
            print(print_str)
            Logger.log_line(print_str, use_timestamp=False)
            return False
        
        TrainedModelMaker.NUM_FILTERS *= 2

        self.ensemble_controller.update_filters_for_binary_models()
        return True
    


if __name__ == "__main__":
    parse_arguments()
    main = Main()
    main.run()