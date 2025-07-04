import csv
import os
from datetime import datetime
from Globals import ModelSettings, EvolutionSettings, DataSettings, LoggingSettings, UniquenessFunctions, FitnessFunctions

INDIVIDUAL_TEMPLATE = {
                "Experiment_ID": 0,
                "Generation": 0,
                "Individual_ID": 0,
                "Individual": 0,
                "Train_Loss": 0,
                "Test_Loss": 0,
                "Precision": 0,
                "Recall": 0,
                "F1": 0,
                "Accuracy": 0,
                "Fitness": 0,
                "Fully_Trained": 0,
                "Uniqueness": 0.0,
                "AlphaBetaFitness": 0.0,
        }

class LogManager:
    """Comprehensive logging system for evolutionary algorithms"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.Experiment_ID = self._get_Experiment_ID()

        self.best_individual_in_generation = INDIVIDUAL_TEMPLATE.copy()
    
    def _get_Experiment_ID(self):
        """Get the next experiment ID based on the CSV log"""

        filepath = self._get_filepath(filetype="Experiment")

        with open(filepath, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            Experiment_IDs = [int(row["Experiment_ID"]) for row in reader if "Experiment_ID" in row and row["Experiment_ID"].isdigit()]
            
            if not Experiment_IDs:
                return 0
            
            return  max(Experiment_IDs) + 1

    def _write_with_config(self, filetype, config):
        
        filepath = self._get_filepath(filetype=filetype)

        with open(filepath, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=config.keys())

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
            "Name": LoggingSettings.experiment_name,
            "Experiment_ID": self.Experiment_ID,
            "Start_Time": self.start_time,
            "End_Time": datetime.now(),
            "Sleepstage": sleepstage,
            "Signal_Type": signal_type,
            "Batch_Size": ModelSettings.BATCH_SIZE,
            "Epochs_Per_Individual": ModelSettings.TRAINING_EPOCHS_PER_INDIVIDUAL,
            "Population_Size": EvolutionSettings.POPULATION_SIZE,
            "Generations": EvolutionSettings.GENERATIONS,
            "Crossover_Prob": EvolutionSettings.CX_PROB,
            "Mutation_Prob": EvolutionSettings.MUTATION_PROB,
            "Tournament_Size": EvolutionSettings.SELECTION_TOURNAMENT_SIZE,
            "Min_Kernel_Size": ModelSettings.MIN_KERNEL_SIZE,
            "Max_Kernel_Size": max_kernel_size,
            "Best": best,
            "Second_Best": second_best,
            "Third_Best": third_best,
            "Dataset_Name": DataSettings.DATASET,
            "Max_Time_Spent_Training": ModelSettings.MAX_TIME_SPENT_TRAINING,
            "Fitness_Function": FitnessFunctions.fitness_function.__name__,
            "Uniqueness_Function": UniquenessFunctions.uniqueness_function.__name__,
            'Alpha': EvolutionSettings.ALPHA_BETA[0],
            'Beta': EvolutionSettings.ALPHA_BETA[1],
            'AB_Switch': EvolutionSettings.BETA_SWITCH
        }

        self._write_with_config(filetype="Experiment", config=config)

    def log_generation_stats(self, population_size:int, mean, std_deviation, median, min, fit_max, test_the_best: bool = False):

        LoggingSettings.current_generation_id
        generation_configs = {
            "Experiment_ID": self.Experiment_ID,
            "Generation": LoggingSettings.current_generation_id,
            "Population_Size": population_size,
            "Fitness_Mean": mean,
            "Fitness_Std": std_deviation,
            "Fitness_Median": median,
            "Fitness_Min": min,
            "Fitness_Max": fit_max,
            "Best_Individual_ID": f"(exp:{self.Experiment_ID},gen:{LoggingSettings.current_generation_id},id:{self.best_individual_in_generation['Individual_ID']}), fitness:{round(self.best_individual_in_generation['Fitness'], 7)}, kernels:{str(self.best_individual_in_generation['Individual'])}",
            "Tournament_Of_Champions": test_the_best
        }

        self._write_with_config(filetype="Generation", config=generation_configs)
        self._write_with_config(filetype="Individual", config=self.best_individual_in_generation)

        self.best_individual_in_generation = INDIVIDUAL_TEMPLATE.copy()

    def check_for_best_in_gen(self, individual):

        fitness = individual.raw_fitness or individual.fitness.values[0]
        uniqueness = individual.uniqueness if hasattr(individual, 'uniqueness') else None
        alpha_beta_fitness = individual.alpha_beta_fitness if hasattr(individual, 'alpha_beta_fitness') else None
        ind_id = individual.individual_id

        train_loss = individual.model_performance.get("Train Loss", 0.0)
        test_loss = individual.model_performance.get("Test Loss", 0.0)
        precision = individual.model_performance.get("Precision", 0.0)
        recall = individual.model_performance.get("Recall", 0.0)
        f1 = individual.model_performance.get("F1", 0.0)
        accuracy = individual.model_performance.get("Accuracy", 0.0)

        best = self.best_individual_in_generation
        generation = LoggingSettings.current_generation_id

        individual_log_entry = {
                "Experiment_ID": self.Experiment_ID,
                "Generation": generation,
                "Individual_ID": ind_id,
                "Individual": str(individual),
                "Train_Loss": round(train_loss, 4),
                "Test_Loss": round(test_loss, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1, 4),
                "Accuracy": round(accuracy, 4),
                "Fitness": round(fitness, 4),
                "Uniqueness": round(uniqueness, 4) if uniqueness else None,
                "AlphaBetaFitness": round(alpha_beta_fitness, 4) if alpha_beta_fitness else None,
        }

        if (best["Fitness"] <= fitness):
            self.best_individual_in_generation = individual_log_entry

        if (LoggingSettings.LOG_ALL_INDIVIDUALS):
            self._write_with_config(filetype="Individual", config=individual_log_entry)
            return
