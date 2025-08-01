import os
import pandas as pd
from datetime import datetime
from Globals import EvolutionManager, LoggingHelper
from ea_controller.fitness_functions import FitnessFunctions
from datahelpers.data import Data

class LogManager:
    """Comprehensive logging system for evolutionary algorithms"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.experiment_id = self._get_id_by("Experiment")
    
    def _get_id_by(self, filetype="Experiment"):
        """Get the next experiment ID based on the CSV log.
        Returns 0 if the log is empty, doesn't exist, or is corrupt.
        """
        try:
            filepath = self._get_filepath(filetype=filetype)
            
            # Check if file exists and has content
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                return 0
                
            df = pd.read_csv(filepath)
            
            if df.empty:
                return 0
                
            if filetype == "Experiment":
                return int(df["experiment_id"].max()) + 1
            # Add handling for other filetypes here if needed
                
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return 0

    def _write_with_config(self, filetype, config):
        filepath = self._get_filepath(filetype=filetype)
        
        # Check if file exists and is empty to determine if we need to write headers
        write_header = not os.path.exists(filepath) or os.stat(filepath).st_size == 0
        
        df = pd.DataFrame([config])

        if write_header:
            df.to_csv(filepath, mode='a', index=False)
        else:
            df.to_csv(filepath, mode='a', header=False, index=False)

    def _get_filepath(self, filetype):

        if filetype == "Experiment":
            inner_path = "_misc/data_from_logs/ExperimentStatsLog.csv"
        elif filetype == "Generation":
            inner_path = f"_misc/data_from_logs/GenerationStatsLog.csv"
        elif filetype == "Individual":
            inner_path = f"_misc/data_from_logs/IndividualLog.csv"
        else:
            raise ValueError(f"Unknown filetype: {filetype}")
        
        SLEAPy_path = f"SLEAP/{inner_path}"

        not_found = []
        check_sleep_path = os.path.isfile(SLEAPy_path)
        if check_sleep_path:
            return SLEAPy_path
        else:
            not_found.append(SLEAPy_path)
        
        check_inner_path = os.path.isfile(inner_path)
        if check_inner_path:
            return inner_path
        else:
            not_found.append(inner_path)
        
        raise FileNotFoundError(f"Could not find file: {not_found}")

    def log_experiment(self, classification_class, signal_type, max_kernel_size, best, second_best, third_best):
        """log the experiment configuration using template names"""
        d = Data()
        
        config = {
            "experiment_id": self.experiment_id,
            "name": LoggingHelper.experiment_name,
            "start_time": self.start_time,
            "end_time": datetime.now(),
            "classification_class": classification_class,
            "signal_type": signal_type,

            "population_size": EvolutionManager.POPULATION_SIZE,
            "generations": EvolutionManager.GENERATIONS,
            "crossover_prob": EvolutionManager.CX_PROB,
            "mutation_prob": EvolutionManager.MUTATION_PROB,

            "max_kernel_size": max_kernel_size,

            "dataset_name": d.dataset,
            "max_memory": d.max_memory,

            "fitness_function": FitnessFunctions.fitness_function.__name__,
            "minimize_fitness": FitnessFunctions.MINIMIZE_FITNESS,
            "best": best,
            "second_best": second_best,
            "third_best": third_best,
        }

        self._write_with_config(filetype="Experiment", config=config)

    def log_generation_stats(self, population, number_of_trained_individual:int, 
                             fit_mean, fit_std_deviation, fit_median, fit_min, fit_max,
                             l_mean, l_std_deviation, l_median, l_min, l_max):


        for indi in population:
            indi_config = self.fill_individual_template(
                generation= LoggingHelper.current_generation_id,
                ind_id= indi.individual_id,
                model_performance= indi.model_performance
            )

            self._write_with_config(filetype="Individual", config=indi_config)

        rounding_number = 2
        generation_configs = {
            "experiment_id": self.experiment_id,
            "generation": LoggingHelper.current_generation_id,
            "number_of_trained_individuals": number_of_trained_individual,
            "fitness_mean": round(fit_mean, rounding_number),
            "fitness_std": round(fit_std_deviation, rounding_number),
            "fitness_median": round(fit_median, rounding_number),
            "fitness_min": round(fit_min, rounding_number),
            "fitness_max": round(fit_max, rounding_number),
            "loss_mean": round(l_mean, rounding_number),
            "loss_std": round(l_std_deviation, rounding_number),
            "loss_median": round(l_median, rounding_number),
            "loss_min": round(l_min, rounding_number),
            "loss_max": round(l_max, rounding_number),
        }

        self._write_with_config(filetype="Generation", config=generation_configs)

    def fill_individual_template(self, generation, ind_id, model_performance):
        """Fill in the individual template with provided values"""
        
        individual_template = {
            "experiment_id": self.experiment_id,
            "generation": generation,
            "indi_id": ind_id,
            "model_performance": {
                k: round(v, 1) if isinstance(v, float) else v 
                for k, v in model_performance.items()
                if isinstance(v, int) or isinstance(v, str) or isinstance(v, float) or isinstance(v, list)
            },
        }
        return individual_template
