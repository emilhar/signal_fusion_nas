import csv
import os
from datetime import datetime
from Globals import ModelSettings, EvolutionSettings, DataSettings

class LogManager:
    """Comprehensive logging system for evolutionary algorithms"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.experiment_id = self._get_experiment_id()

        self.current_generation_id = 0
        self.current_individual_id = -1

        self.best_individual_in_generation = {
                "experiment_id": 0,
                "generation": 0,
                "individual_id": 0,
                "individual": 0,
                "Train Loss": 0,
                "Test Loss": 0,
                "Precision": 0,
                "Recall": 0,
                "F1": 0,
                "Accuracy": 0,
                "fitness": 0,
                "champion": 0,
        }
    
    def _get_experiment_id(self):
        """Get the next experiment ID based on the CSV log"""

        filepath = self._get_filepath(filetype="Experiment")

        with open(filepath, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            experiment_ids = [int(row["experiment_id"]) for row in reader if "experiment_id" in row and row["experiment_id"].isdigit()]
            
            if not experiment_ids:
                return 0
            
            return  max(experiment_ids) + 1

    def _write_with_config(self, filetype, config):
        
        filepath = self._get_filepath(filetype=filetype)

        with open(filepath, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=config.keys())

            writer.writerow(config)

    def _get_filepath(self, filetype):

        if filetype == "Experiment":
            inner_path = "Logs/ExperimentStatsLog.csv"
        elif filetype == "Generation":
            inner_path = "Logs/GenerationStatsLog.csv"
        elif filetype == "Individual":
            inner_path = "Logs/IndividualLog.csv"
        elif filetype == "HallOfFame":
            inner_path = "Logs/HallOfFameLog.csv"
        else:
            raise ValueError(f"Unknown filetype: {filetype}")
        
        SLEAP_path = f"SLEAP/{inner_path}"

        check_sleep_path = os.path.isfile(SLEAP_path)
        if check_sleep_path:
            return SLEAP_path
        
        check_inner_path = os.path.isfile(inner_path)
        if check_inner_path:
            return inner_path
                
        raise FileNotFoundError(f"Could not find file: {SLEAP_path}")

    def log_experiment(self, sleepstage, signal_type, max_kernel_size, best, second_best, third_best):
        """Log the experiment configuration"""

        config = {
            "experiment_id": self.experiment_id,
            "start_time":  self.start_time,
            "end_time": datetime.now(),
            "sleepstage": sleepstage,
            "signal_type": signal_type,
            "batch_size": ModelSettings.BATCH_SIZE,
            "epochs_per_individual": ModelSettings.TRAINING_EPOCHS_PER_INDIVIDUAL,
            "population_size": EvolutionSettings.POPULATION_SIZE,
            "generations": EvolutionSettings.GENERATIONS,
            "offspring_variation": EvolutionSettings.OFFSPRING_VARIATION,
            "crossover_prob": EvolutionSettings.CX_PROB,
            "mutation_prob": EvolutionSettings.MUTATION_PROB,
            "tournament_size": EvolutionSettings.TOURNAMENT_SIZE,
            "min_kernel_size": ModelSettings.MIN_KERNEL_SIZE,
            "max_kernel_size": max_kernel_size,
            "data_points_per_individual": EvolutionSettings.DATA_POINTS_PER_INDIVIUAL,
            "best": best,
            "second_best": second_best,
            "third_best": third_best,
            "toc_on": EvolutionSettings.TOC_ON,
            "toc_generations_between": EvolutionSettings.TOC_GENERATIONS_BETWEEN,
            "toc_tournament_size": EvolutionSettings.TOC_TOURNAMENT_SIZE,
            "dataset_name": DataSettings.DATASET,
            "max_time_spent_training": ModelSettings.MAX_TIME_SPENT_TRAINING,
            "toc_batch_size": EvolutionSettings.TOC_BATCH_SIZE,
            "fitness_function": EvolutionSettings.FITNESS_FUNCTION
        }

        self._write_with_config(filetype="Experiment", config=config)

    def log_hall_of_fame(self, hall_of_fame_list):
        config = {"experiment_id": self.experiment_id}

        for i, member in enumerate(hall_of_fame_list):
            config[f"Nr.{i+1}"] = member

        self._write_with_config(filetype="HallOfFame", config=config)


    def log_generation_stats(self, generation: int, population_size:int, mean, std_deviation, median, min, fit_max, tournament_of_champions: bool = False):

        self.current_generation_id = generation

        generation_configs = {
            "experiment_id": self.experiment_id,
            "generation": self.current_generation_id,
            "population_size": population_size,
            "fitness_mean": mean,
            "fitness_std": std_deviation,
            "fitness_median": median,
            "fitness_min": min,
            "fitness_max": fit_max,
            "best_individual_id": f"(exp:{self.experiment_id},gen:{self.current_generation_id},id:{self.best_individual_in_generation["individual_id"]}), fitness:{self.best_individual_in_generation["fitness"]}, kernels:{str(self.best_individual_in_generation["individual"])}",
            "tournament_of_champions": tournament_of_champions}

        self._write_with_config(filetype="Generation", config=generation_configs)

        self._write_with_config(filetype="Individual", config=self.best_individual_in_generation)

        self.current_generation_id = generation + 1
        self.current_individual_id = -1

        self.best_individual_in_generation = {
                "experiment_id": 0,
                "generation": 0,
                "individual_id": 0,
                "individual": 0,
                "Train Loss": 0,
                "Test Loss": 0,
                "Precision": 0,
                "Recall": 0,
                "F1": 0,
                "Accuracy": 0,
                "fitness": 0,
                "champion": 0,
        }

    def check_for_best_in_gen(self, individual, fitness, champion, train_loss, test_loss, precision, recall, f1, accuracy):

        train_loss, test_loss, precision, recall, f1, accuracy = map(lambda x: x[0], [train_loss, test_loss, precision, recall, f1, accuracy])

        self.current_individual_id += 1
        best = self.best_individual_in_generation
        generation = self.current_generation_id
        
        if (best["fitness"] <= fitness):

            self.best_individual_in_generation = {
                "experiment_id": self.experiment_id,
                "generation": generation,
                "individual_id": self.current_individual_id,
                "individual": str(individual),
                "Train Loss": train_loss,
                "Test Loss": test_loss,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Accuracy": accuracy,
                "fitness": fitness,
                "champion": champion,
        }

        if (champion):
            self.log_individual_stats(individual, fitness, train_loss, test_loss, precision, recall, f1, accuracy, champion)
            return

    def log_individual_stats(self, individual, fitness, train_loss, test_loss, precision, recall, f1, accuracy, champion: bool = False):
        """Log individual evaluation"""

        # Make entry
        generation = self.current_generation_id
        individual_log_entry = {
                "experiment_id": self.experiment_id,
                "generation": generation,
                "individual_id": self.current_individual_id,
                "individual": str(individual),
                "Train Loss": train_loss,
                "Test Loss": test_loss,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Accuracy": accuracy,
                "fitness": fitness,
                "champion": champion,
        }
        
        self._write_with_config(filetype="Individual", config=individual_log_entry)
