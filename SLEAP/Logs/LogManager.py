import os
import pandas as pd
from datetime import datetime
from Globals import EvolutionManager, LoggingSettings, FitnessFunctions, LoggingTemplate, DataManager
from ea_controller.ea_controller import KernelSizeEvolutionaryOptimizer

class LogManager:
    """Comprehensive logging system for evolutionary algorithms"""
    
    def __init__(self):
        self.lt = LoggingTemplate
        self.start_time = datetime.now()
        self.Experiment_ID = self._get_id_by("Experiment")
    
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
                return int(df[self.lt.experiment_id].max()) + 1
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

    def log_experiment(self, classification_class, signal_type, max_kernel_size, best, second_best, third_best):
        """Log the experiment configuration using template names"""
        lt = LoggingTemplate
        
        config = {
            lt.experiment_id: self.Experiment_ID,
            "name": LoggingSettings.experiment_name,
            "start_time": self.start_time,
            "end_time": datetime.now(),
            lt.classification_class: classification_class,
            lt.signal_type: signal_type,

            lt.population_size: EvolutionManager.POPULATION_SIZE,
            lt.generations: EvolutionManager.GENERATIONS,
            lt.crossover_prob: EvolutionManager.CX_PROB,
            lt.mutation_prob: EvolutionManager.MUTATION_PROB,
            lt.selection_tournament_size: EvolutionManager.SELECTION_TOURNAMENT_SIZE,
            lt.elitism: EvolutionManager.ELITISM,
            lt.hall_of_fame_members: EvolutionManager.HALL_OF_FAME_MEMBERS,
            lt.max_number_of_mutations: EvolutionManager.MAX_NUMBER_OF_MUTATIONS,
            lt.data_split_training: EvolutionManager.DATA_SPLIT_TRAINING,
            lt.data_split_testing: EvolutionManager.DATA_SPLIT_TESTING,

            lt.epoch: KernelSizeEvolutionaryOptimizer.TRAINING_EPOCHS_PER_INDIVIDUAL,
            lt.min_kernel_size: KernelSizeEvolutionaryOptimizer.MIN_KERNEL_SIZE,
            lt.max_kernel_size: max_kernel_size,
            lt.number_of_branches_range: KernelSizeEvolutionaryOptimizer.NUMBER_OF_BRANCHES_RANGE,
            lt.number_of_kernels_range: KernelSizeEvolutionaryOptimizer.NUMBER_OF_KERNELS_RANGE,

            lt.dataset_name: DataManager.DATASET,
            lt.max_memory: DataManager.MAX_MEMORY,

            lt.fitness_function: FitnessFunctions.fitness_function.__name__,
            lt.minimize_fitness: FitnessFunctions.MINIMIZE_FITNESS,
            "best": best,
            "second_best": second_best,
            "third_best": third_best,
        }

        self._write_with_config(filetype="Experiment", config=config)

    def log_generation_stats(self, population, number_of_trained_individual:int, 
                             fit_mean, fit_std_deviation, fit_median, fit_min, fit_max,
                             l_mean, l_std_deviation, l_median, l_min, l_max):

        if not LoggingSettings.LOG_ALL_INDIVIDUALS:
            # Log the best individual in the population

            best = population[0]
            for indi in population:
                if FitnessFunctions.MINIMIZE_FITNESS:
                    if indi.fitness.values[0] < best.fitness.values[0]:
                        best = indi
                else:
                    if indi.fitness.values[0] > best.fitness.values[0]:
                        best = indi

            best_config = self.fill_individual_template(
                generation= LoggingSettings.current_generation_id,
                ind_id= best.individual_id,
                model_performance= best.model_performance
            )
            self._write_with_config(filetype="Individual", config=best_config)
        
        else:
            for indi in population:
                indi_config = self.fill_individual_template(
                    generation= LoggingSettings.current_generation_id,
                    ind_id= indi.individual_id,
                    model_performance= indi.model_performance
                )

                self._write_with_config(filetype="Individual", config=indi_config)

        generation_configs = {
            self.lt.experiment_id: self.Experiment_ID,
            self.lt.generation: LoggingSettings.current_generation_id,
            "number_of_trained_individuals": number_of_trained_individual,
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

    def fill_individual_template(self, generation, ind_id, model_performance):
        """Fill in the individual template with provided values"""
        
        individual_template = {
            self.lt.experiment_id: self.Experiment_ID,
            self.lt.generation: generation,
            self.lt.indi_id: ind_id,
            "model_performance": {
                k: round(v, self.lt.rounding_number) if isinstance(v, float) else v 
                for k, v in model_performance.items()
                if isinstance(v, int) or isinstance(v, str) or isinstance(v, float) or isinstance(v, list)
            },
        }
        return individual_template
