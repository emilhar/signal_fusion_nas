import csv
import os
from datetime import datetime
import pandas as pd
from Globals import ModelManager, EvolutionManager, DataManager, LoggingSettings, AlpsManager, FitnessFunctions, LoggingTemplate

class LogManager:
    """Comprehensive logging system for evolutionary algorithms"""
    
    def __init__(self):
        self.lt = LoggingTemplate()
        self.start_time = datetime.now()
        self.Experiment_ID = self._get_id_by("Experiment")
    
    def _get_id_by(self, filetype="Experiment"):
        """Get the next experiment ID based on the CSV log"""

        filepath = self._get_filepath(filetype=filetype)
        df = pd.read_csv(filepath)
        if df.empty:
            return 0
        
        if filetype == "Experiment":
            return df[self.lt.experiment_id].max() + 1

    def _write_with_config(self, filetype, config):
        filepath = self._get_filepath(filetype=filetype)
        
        # Check if file exists and is empty to determine if we need to write headers
        write_header = not os.path.exists(filepath) or os.stat(filepath).st_size == 0
        
        with open(filepath, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=config.keys())
            
            # Write header only if file is new/empty
            if write_header:
                writer.writeheader()
                
            writer.writerow(config)

    def _get_filepath(self, filetype):

        if filetype == "Experiment":
            inner_path = f"Logs/{LoggingSettings.LOGGER_ID}Logs/ExperimentStatsLog.csv"
        elif filetype == "Generation":
            inner_path = f"Logs/{LoggingSettings.LOGGER_ID}Logs/GenerationStatsLog.csv"
        elif filetype == "Individual":
            inner_path = f"Logs/{LoggingSettings.LOGGER_ID}Logs/IndividualLog.csv"
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

    def log_experiment(self, sleepstage, signal_type, max_kernel_size, best, second_best, third_best):
        """Log the experiment configuration using template names"""

        config = {
            self.lt.experiment_id: self.Experiment_ID,
            "name": LoggingSettings.experiment_name,
            "start_time": self.start_time,
            "end_time": datetime.now(),
            "sleepstage": sleepstage,
            "signal_type": signal_type,
            "base_batch_size": ModelManager.BATCH_SIZE,
            "population_size": EvolutionManager.POPULATION_SIZE_PER_LAYER,
            "generations": EvolutionManager.GENERATIONS,
            "crossover_prob": EvolutionManager.CX_PROB,
            "mutation_prob": EvolutionManager.MUTATION_PROB,
            "selection_tournament_size": EvolutionManager.SELECTION_TOURNAMENT_SIZE,
            "min_kernel_size": ModelManager.MIN_KERNEL_SIZE,
            "max_kernel_size": max_kernel_size,
            "best": best,
            "second_best": second_best,
            "third_best": third_best,
            "dataset_name": DataManager.DATASET,
            "fitness_function": FitnessFunctions.fitness_function.__name__,
            "age_gap": AlpsManager.AGE_GAP,
            "aging_scheme": AlpsManager.AgingScheme.uas_str,
            "alps_Manager": AlpsManager.TRAINING_SETTINGS_FOR_LAYERS.__repr__()
        }

        self._write_with_config(filetype="Experiment", config=config)

    def log_generation_stats(self, population, number_of_trained_individual:int, 
                             fit_mean, fit_std_deviation, fit_median, fit_min, fit_max,
                             l_mean, l_std_deviation, l_median, l_min, l_max):

        if not LoggingSettings.LOG_ALL_INDIVIDUALS:
            layers = {}
            for ind in population:
                if ind.layer not in layers:
                    layers[ind.layer] = []
                layers[ind.layer].append(ind)
            
            people_in_layers_count = [0 for _ in AlpsManager.teitur_percentages]

            for layer, layer_population in layers.items():
                people_in_layers_count[layer] = (len(layer_population))
                best = max(layer_population, key=lambda x: x.fitness.values[0])
                
                best_in_layer = self.fill_individual_template(
                    generation= LoggingSettings.current_generation_id,
                    ind_id= best.individual_id,
                    age= best.age,
                    layer= best.layer,
                    model_performance= best.model_performance
                )

                self._write_with_config(filetype="Individual", config=best_in_layer)

        else:
            people_in_layers_count = [0 for _ in AlpsManager.teitur_percentages]
            for indi in population:
                people_in_layers_count[indi.layer]+=1
                indi_config = self.fill_individual_template(
                    generation= LoggingSettings.current_generation_id,
                    ind_id= indi.individual_id,
                    age= indi.age,
                    layer= indi.layer,
                    model_performance= indi.model_performance
                )

                self._write_with_config(filetype="Individual", config=indi_config)

        generation_configs = {
            self.lt.experiment_id: self.Experiment_ID,
            self.lt.generation: LoggingSettings.current_generation_id,
            "number_of_trained_individuals": number_of_trained_individual,
            "individual_count_per_layer": people_in_layers_count,
            "fitness_mean": round(fit_mean, self.lt.rounding_number),
            "fitness_std": round(fit_std_deviation, self.lt.rounding_number),
            "fitness_median": round(fit_median, self.lt.rounding_number),
            "fitness_min": round(fit_min, self.lt.rounding_number),
            "fitness_max": round(fit_max, self.lt.rounding_number),
            "loss_mean": round(l_mean, self.lt.rounding_number),
            "loss_std": round(l_std_deviation, self.lt.rounding_number),
            "loss_median": round(l_median, self.lt.rounding_number),
            "loss_min": round(l_min, self.lt.rounding_number),
            "loss_max": round(l_max, self.lt.rounding_number),
        }

        self._write_with_config(filetype="Generation", config=generation_configs)

    def fill_individual_template(self, generation, ind_id, age, layer, 
                               model_performance):
        """Fill in the individual template with provided values"""
        
        individual_template = {
            self.lt.experiment_id: self.Experiment_ID,
            self.lt.generation: generation,
            self.lt.indi_id: ind_id,
            self.lt.age: age,
            self.lt.layer: layer,
            "model_performance": {
                k: round(v, self.lt.rounding_number) if isinstance(v, float) else v 
                for k, v in model_performance.items()
                if isinstance(v, int) or isinstance(v, str) or isinstance(v, float) or isinstance(v, list)
            },
        }
        return individual_template
