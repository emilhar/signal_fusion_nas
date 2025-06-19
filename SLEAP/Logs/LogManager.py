EXPERIMENT_STATS_LOG_PATH = "SLEAP/Logs/ExperimentStatsLog.csv"
GENERATION_STATS_LOG_PATH = "SLEAP/Logs/GenerationStatsLog.csv"
INDIVIDUAL_STATS_LOG_PATH = "SLEAP/Logs/IndividualLog.csv"

import csv
import os
from datetime import datetime
from Globals import ModelSettings, EvolutionSettings, DataSettings

class LogManager:
    """Comprehensive logging system for evolutionary algorithms"""
    
    def __init__(self):
        self.start_time = datetime.now()

        self.current_generation_id = 0
        self.current_individual_id = -1

        self.best_individual_in_generation = {
            "generation": -1,
            "fitness": -1,
            "individual": None,
            "champion": False,
            'train_loss': 0, 
            'test_loss': 0,
            'precision':0,
            'recall': 0, 
            'f1': 0,
            'accuracy': 0}

        self.experiment_id = self._get_experiment_id()

    def _get_experiment_id(self):
        """Get the next experiment ID based on the CSV log"""

        if not os.path.isfile(EXPERIMENT_STATS_LOG_PATH):
            return 0

        with open(EXPERIMENT_STATS_LOG_PATH, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            experiment_ids = [int(row["experiment_id"]) for row in reader if "experiment_id" in row and row["experiment_id"].isdigit()]
            
            if not experiment_ids:
                return 0
            
            return  max(experiment_ids) + 1

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
            "dataset_fraction":  ModelSettings.DATASET_FRACTION,
            "population_size": EvolutionSettings.POPULATION_SIZE,
            "generations": EvolutionSettings.GENERATIONS,
            "crossover_prob": EvolutionSettings.CX_PROB,
            "mutation_prob": EvolutionSettings.MUTATION_PROB,
            "tournament_size": EvolutionSettings.TOURNAMENT_SIZE,
            "min_kernel_size": ModelSettings.MIN_KERNEL_SIZE,
            "max_kernel_size": max_kernel_size,
            "best": best,
            "second_best": second_best,
            "third_best": third_best,
            "toc_on": EvolutionSettings.TOC_ON,
            "toc_generations_between": EvolutionSettings.TOC_GENERATIONS_BETWEEN,
            "toc_tournament_size": EvolutionSettings.TOC_TOURNAMENT_SIZE,
            "dataset_name": DataSettings.DATASET
        }
        
        file_exists = os.path.isfile(EXPERIMENT_STATS_LOG_PATH)

        with open(EXPERIMENT_STATS_LOG_PATH, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=config.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(config)

    def check_for_best_in_gen(self, individual, fitness, champion, train_loss, test_loss, precision, recall, f1, accuracy):

        train_loss, test_loss, precision, recall, f1, accuracy = map(lambda x: x[0], [train_loss, test_loss, precision, recall, f1, accuracy])

        self.current_individual_id += 1
        best = self.best_individual_in_generation
        generation = self.current_generation_id
        
        if (
            not champion and 
            (
                (best["generation"] < generation) 
                or 
                (best["generation"] == generation and best["fitness"] < fitness)
            )
        ):
            self.best_individual_in_generation = {
                "generation": generation,
                "fitness": fitness,
                "individual": individual,
                "champion": champion,
                'train_loss': train_loss, 
                'test_loss': test_loss,
                'precision': precision,
                'recall': recall, 
                'f1': f1,
                'accuracy': accuracy
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
        
        # Write entry
        file_exists = os.path.exists(INDIVIDUAL_STATS_LOG_PATH)
        
        with open(INDIVIDUAL_STATS_LOG_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=individual_log_entry.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(individual_log_entry)

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
            "best_individual_id": str(self.best_individual_in_generation),
            "tournament_of_champions": tournament_of_champions}
        

        file_exists = os.path.exists(GENERATION_STATS_LOG_PATH)
        
        with open(GENERATION_STATS_LOG_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=generation_configs.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(generation_configs)

        self.current_individual_id = -1
        self.current_generation_id = generation + 1