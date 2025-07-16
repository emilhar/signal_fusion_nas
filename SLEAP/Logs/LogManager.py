import csv
import os
from datetime import datetime
import pandas as pd
from Globals import ModelManager, EvolutionManager, DataManager, LoggingSettings, AlpsManager, FitnessFunctions, LoggingTemplate, PolyarithmosManager

class LogManager:
    """Comprehensive logging system for evolutionary algorithms"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.Experiment_ID = self._get_id_by("Experiment")
        self.best_individual_in_generation = self.fill_individual_template(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        self.lt = LoggingTemplate()
    
    def _get_id_by(self, filetype="Experiment"):
        """Get the next experiment ID based on the CSV log"""

        filepath = self._get_filepath(filetype=filetype)
        df = pd.read_csv(filepath)
        if df.empty:
            return 0
        
        if filetype == "Experiment":
            return df[self.lt.experiment_id].max() + 1
        elif filetype == "Polyarithmos":
            return df[self.lt.PID].max() + 1

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

        if filetype == "Polyarithmos":
            inner_path = f"Logs/{LoggingSettings.LOGGER_ID}Logs/PolyarithmosLog.csv"
        elif filetype == "Experiment":
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

    def log_polyarithmos(self):
        PID = self._get_id_by("Polyarithmos")
        config = {
            self.lt.PID: PID,
            self.lt.experiment_ids_within_polyarithmos: PolyarithmosManager.experiment_ids_within_polyartihmos,
        }
        self._write_with_config(filetype="Polyarithmos", config=config)

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
            "alps_Manager": AlpsManager.TRAINING_Manager_FOR_LAYERS.__repr__()
        }

        self._write_with_config(filetype="Experiment", config=config)

    def log_generation_stats(self, population, number_of_trained_individual:int, mean, std_deviation, median, min, fit_max):

        # list of an amount of zeroes equal to the number of layers, 6 layers = [0,0,0,0,0,0]
        # used for indexing in the for loop
        people_in_layers_count  = [0] * len(AlpsManager.MAX_AGE_IN_LAYERS)
        for person in population:
            people_in_layers_count[person.layer] += 1
        

        generation_configs = {
        self.lt.experiment_id: self.Experiment_ID,
        self.lt.generation: LoggingSettings.current_generation_id,
        "number_of_trained_individuals": number_of_trained_individual,
        "individual_count_per_layer": people_in_layers_count,
        "fitness_mean": round(mean, self.lt.rounding_number),
        "fitness_std": round(std_deviation, self.lt.rounding_number),
        "fitness_median": round(median, self.lt.rounding_number),
        "fitness_min": round(min, self.lt.rounding_number),
        "fitness_max": round(fit_max, self.lt.rounding_number),
        "best_individual_info": f"(exp:{self.Experiment_ID},gen:{LoggingSettings.current_generation_id},id:{self.best_individual_in_generation[self.lt.indi_id]}), fitness:{round(self.best_individual_in_generation[self.lt.fitness], 7)}, branches:{str(self.best_individual_in_generation[self.lt.branches])}",
        }

        self._write_with_config(filetype="Generation", config=generation_configs)

        if not LoggingSettings.LOG_ALL_INDIVIDUALS:
            
            # Log the best individual in the generation
            self._write_with_config(filetype="Individual", config=self.best_individual_in_generation)

            # Log the best individual in each layer
            population_grouped_by_layer = [[]] * len(AlpsManager.MAX_AGE_IN_LAYERS)
            for individual in population:
                population_grouped_by_layer[individual.layer].append(individual)

            for layer_population in population_grouped_by_layer:
                best_in_layer = (0,0)
                for individual in layer_population:
                    if individual.fitness.values[0] > best_in_layer[1]:
                        best_in_layer = (individual, individual.fitness.values[0])
                
                individual = best_in_layer[0]
                best_in_layer = self.fill_individual_template(
                    generation= LoggingSettings.current_generation_id,
                    ind_id= individual.individual_id,
                    individual=str(individual),
                    age= individual.age,
                    layer= individual.layer,
                    train_loss= individual.model_performance.get(self.lt.train_loss, 0.0),
                    test_loss= individual.model_performance.get(self.lt.test_loss, 0.0),
                    precision= individual.model_performance.get(self.lt.precision, 0.0),
                    recall= individual.model_performance.get(self.lt.recall, 0.0),
                    f1= individual.model_performance.get(self.lt.best_f1, 0.0),
                    accuracy= individual.model_performance.get(self.lt.accuracy, 0.0),
                    fitness= individual.fitness.values[0],
                    reason= LoggingTemplate.reason_map["Best In Layer"]
                )
                self._write_with_config(filetype="Individual", config= best_in_layer)

        self.best_individual_in_generation = self.fill_individual_template(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def check_for_best_in_gen(self, individual):
        fitness = individual.fitness.values[0]
        ind_id = individual.individual_id

        train_loss= individual.model_performance.get(self.lt.train_loss, 0.0)
        test_loss= individual.model_performance.get(self.lt.test_loss, 0.0)
        precision= individual.model_performance.get(self.lt.precision, 0.0)
        recall= individual.model_performance.get(self.lt.recall, 0.0)
        f1= individual.model_performance.get(self.lt.best_f1, 0.0)
        accuracy= individual.model_performance.get(self.lt.accuracy, 0.0)
        fitness= individual.fitness.values[0]


        generation = LoggingSettings.current_generation_id

        individual_log_entry = self.fill_individual_template(
            generation=generation,
            ind_id=ind_id,
            individual=str(individual),
            age=individual.age,
            layer=individual.layer,
            train_loss=train_loss,
            test_loss=test_loss,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            fitness=fitness,
            reason=self.lt.reason_map["Checked For Best In Generation"]
        )

        best = self.best_individual_in_generation
        if (best[self.lt.fitness] <= fitness):
            self.best_individual_in_generation = individual_log_entry

        if (LoggingSettings.LOG_ALL_INDIVIDUALS):
            self._write_with_config(filetype="Individual", config=individual_log_entry)
            return

    def fill_individual_template(self, generation, ind_id, individual, age, layer, 
                               train_loss, test_loss, precision, recall, f1, accuracy, fitness, reason):
        """Fill in the individual template with provided values"""

        try:
            individual_template = {
                self.lt.experiment_id: self.Experiment_ID,
                self.lt.generation: generation,
                self.lt.indi_id: ind_id,
                self.lt.branches: str(individual),
                self.lt.age: age,
                self.lt.layer: layer,
                self.lt.train_loss: round(train_loss, self.lt.rounding_number),
                self.lt.test_loss: round(test_loss, self.lt.rounding_number),
                self.lt.precision: round(precision, self.lt.rounding_number),
                self.lt.recall: round(recall, self.lt.rounding_number),
                self.lt.best_f1: round(f1, self.lt.rounding_number),
                self.lt.accuracy: round(accuracy, self.lt.rounding_number),
                self.lt.fitness: round(fitness, self.lt.rounding_number),
                self.lt.reason: reason
            }
            return individual_template
        except TypeError as e:
            print(train_loss)
