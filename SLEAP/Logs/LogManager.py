import csv
import os
from datetime import datetime
from Globals import ModelSettings, EvolutionSettings, DataSettings, LoggingSettings, AlpsSettings, FitnessFunctions, LoggingTemplate

class LogManager:
    """Comprehensive logging system for evolutionary algorithms"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.Experiment_ID = self._get_Experiment_ID()

        self.best_individual_in_generation = self.fill_individual_template(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _get_Experiment_ID(self):
        """Get the next experiment ID based on the CSV log"""

        filepath = self._get_filepath(filetype="Experiment")

        with open(filepath, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            Experiment_IDs = [int(row["Experiment_ID"]) for row in reader if "Experiment_ID" in row and row["Experiment_ID"].isdigit()]
            
            if not Experiment_IDs:
                return 0
            
            return max(Experiment_IDs) + 1

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
        
        SLEAP_path = f"SLEAP/{inner_path}"

        not_found = []
        check_sleep_path = os.path.isfile(SLEAP_path)
        if check_sleep_path:
            return SLEAP_path
        else:
            not_found.append(SLEAP_path)
        
        check_inner_path = os.path.isfile(inner_path)
        if check_inner_path:
            return inner_path
        else:
            not_found.append(inner_path)
        
        raise FileNotFoundError(f"Could not find file: {not_found}")

    def log_experiment(self, sleepstage, signal_type, max_kernel_size, best, second_best, third_best):
        """Log the experiment configuration using template names"""
        it = LoggingTemplate()

        config = {
            it.experiment_id: self.Experiment_ID,
            "name": LoggingSettings.experiment_name,
            "start_time": self.start_time,
            "end_time": datetime.now(),
            "sleepstage": sleepstage,
            "signal_type": signal_type,
            "base_batch_size": ModelSettings.BATCH_SIZE,
            "population_size": EvolutionSettings.POPULATION_SIZE_PER_LAYER,
            "generations": EvolutionSettings.GENERATIONS,
            "crossover_prob": EvolutionSettings.CX_PROB,
            "mutation_prob": EvolutionSettings.MUTATION_PROB,
            "selection_tournament_size": EvolutionSettings.SELECTION_TOURNAMENT_SIZE,
            "min_kernel_size": ModelSettings.MIN_KERNEL_SIZE,
            "max_kernel_size": max_kernel_size,
            "best": best,
            "second_best": second_best,
            "third_best": third_best,
            "dataset_name": DataSettings.DATASET,
            "max_time_on": ModelSettings.HAVE_MAX_TIME,
            "max_time_spent_training": ModelSettings.MAX_TIME_SPENT_TRAINING,
            "fitness_function": FitnessFunctions.fitness_function.__name__,
            "age_gap": AlpsSettings.AGE_GAP,
            "aging_scheme": AlpsSettings.AgingScheme.uas_str,
            "alps_settings": AlpsSettings.TRAINING_SETTINGS_FOR_LAYERS.__repr__()
        }

        self._write_with_config(filetype="Experiment", config=config)

    def log_generation_stats(self, population, number_of_trained_individual:int, mean, std_deviation, median, min, fit_max):

        # list of an amount of zeroes equal to the number of layers, 6 layers = [0,0,0,0,0,0]
        # used for indexing in the for loop
        people_in_layers_count  =[0] * len(AlpsSettings.LAYER_CREATION_THRESHOLDS)
        for person in population:
            people_in_layers_count[person.layer] += 1
        
        it = LoggingTemplate()

        generation_configs = {
        it.experiment_id: self.Experiment_ID,
        it.generation: LoggingSettings.current_generation_id,
        "number_of_trained_individuals": number_of_trained_individual,
        "individual_count_per_layer": people_in_layers_count,
        "fitness_mean": round(mean, it.rounding_number),
        "fitness_std": round(std_deviation, it.rounding_number),
        "fitness_median": round(median, it.rounding_number),
        "fitness_min": round(min, it.rounding_number),
        "fitness_max": round(fit_max, it.rounding_number),
        "best_individual_info": f"(exp:{self.Experiment_ID},gen:{LoggingSettings.current_generation_id},id:{self.best_individual_in_generation[it.indi_id]}), fitness:{round(self.best_individual_in_generation[it.fitness], 7)}, branches:{str(self.best_individual_in_generation[it.branches])}",
        }

        self._write_with_config(filetype="Generation", config=generation_configs)

        if not LoggingSettings.LOG_ALL_INDIVIDUALS:
            
            # Log the best individual in the generation
            self._write_with_config(filetype="Individual", config=self.best_individual_in_generation)

            # Log the best individual in each layer
            population_grouped_by_layer = [[]] * len(AlpsSettings.LAYER_CREATION_THRESHOLDS)
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
                    train_loss= individual.model_performance.get(it.train_loss, 0.0),
                    test_loss= individual.model_performance.get(it.test_loss, 0.0),
                    precision= individual.model_performance.get(it.precision, 0.0),
                    recall= individual.model_performance.get(it.recall, 0.0),
                    f1= individual.model_performance.get(it.best_f1, 0.0),
                    accuracy= individual.model_performance.get(it.accuracy, 0.0),
                    fitness= individual.fitness.values[0],
                    reason= LoggingTemplate.reason_map["Best In Layer"]
                )
                self._write_with_config(filetype="Individual", config= best_in_layer)

        self.best_individual_in_generation = self.fill_individual_template(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def check_for_best_in_gen(self, individual):
        fitness = individual.fitness.values[0]
        ind_id = individual.individual_id

        it = LoggingTemplate()
        train_loss= individual.model_performance.get(it.train_loss, 0.0)
        test_loss= individual.model_performance.get(it.test_loss, 0.0)
        precision= individual.model_performance.get(it.precision, 0.0)
        recall= individual.model_performance.get(it.recall, 0.0)
        f1= individual.model_performance.get(it.best_f1, 0.0)
        accuracy= individual.model_performance.get(it.accuracy, 0.0)
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
            reason=it.reason_map["Checked For Best In Generation"]
        )

        best = self.best_individual_in_generation
        if (best[it.fitness] <= fitness):
            self.best_individual_in_generation = individual_log_entry

        if (LoggingSettings.LOG_ALL_INDIVIDUALS):
            self._write_with_config(filetype="Individual", config=individual_log_entry)
            return

    def fill_individual_template(self, generation, ind_id, individual, age, layer, 
                               train_loss, test_loss, precision, recall, f1, accuracy, fitness, reason):
        """Fill in the individual template with provided values"""

        it = LoggingTemplate()
        try:
            individual_template = {
                it.experiment_id: self.Experiment_ID,
                it.generation: generation,
                it.indi_id: ind_id,
                it.branches: str(individual),
                it.age: age,
                it.layer: layer,
                it.train_loss: round(train_loss, it.rounding_number),
                it.test_loss: round(test_loss, it.rounding_number),
                it.precision: round(precision, it.rounding_number),
                it.recall: round(recall, it.rounding_number),
                it.best_f1: round(f1, it.rounding_number),
                it.accuracy: round(accuracy, it.rounding_number),
                it.fitness: round(fitness, it.rounding_number),
                it.reason: reason
            }
            return individual_template
        except TypeError as e:
            print(train_loss)
