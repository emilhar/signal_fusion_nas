import numpy as np
import random
import time
from Globals import EvolutionManager, LoggingSettings, LoggingTemplate, Clr


class SLeaMuPlusLambda:

    def __init__(self, population, toolbox, halloffame, LogManager):
        self.population = population
        self.toolbox = toolbox
        self.mu = EvolutionManager.POPULATION_SIZE
        self.lambda_ = EvolutionManager.POPULATION_SIZE // 2
        self.halloffame = halloffame
        self.LogManager = LogManager 

    def eaMuPlusLambda(self, stats=None):
        r"""
        Modified eaMuPlusLambda comes from the deap.algorithms.eaSimple that can be further seen here:
        https://github.com/DEAP/deap/tree/master
        """
        if EvolutionManager.VERBOSE: print("=== FIRST GENERATION ===")
        if not EvolutionManager.VERBOSE:
            self._loading_bar(-1, 0, 0)

        gen0_start_time = time.time()

        # Get all individuals without fitness values
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]

        # Save info to LoggingSettings to help verbosity
        LoggingSettings.population_size = len(invalid_ind)
        LoggingSettings.current_generation_id = 0

        # Evaluate the individuals with an invalid fitness
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update Hall Of Fame
        if self.halloffame is not None:
            self.halloffame.update(self.population)

        self.end_generation(stats, invalid_ind)
        gen0_time = time.time() - gen0_start_time
        max_time_per_gen = gen0_time
        elapsed_time = gen0_time

        if not EvolutionManager.VERBOSE:
            self._loading_bar(0, max_time_per_gen, elapsed_time)


        # Begin the generational process
        for gen in range(1, EvolutionManager.GENERATIONS + 1):

            gen_start_time = time.time()

            if EvolutionManager.VERBOSE: 
                print("\n=== NEW GENERATION ===")
            else:
                self._loading_bar(gen, max_time_per_gen, elapsed_time)

            # Save info to LoggingSettings to help verbosity 
            LoggingSettings.population_size = len(self.population)
            LoggingSettings.current_generation_id = gen
            LoggingSettings.current_individual_id = 0

            # Vary the population
            offspring = self.varOr()

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if self.halloffame is not None:
                self.halloffame.update(offspring)

            # Select the next generation population
            self.population[:] = self.toolbox.select(self.population + offspring, self.mu)

            # Update the statistics with the new population
            self.end_generation(stats, invalid_ind)
            
            # Update max time per generation
            gen_time = time.time() - gen_start_time
            if gen_time > max_time_per_gen:
                max_time_per_gen = gen_time


        return self.population
    
    def varOr(self):

        assert (EvolutionManager.CX_PROB + EvolutionManager.MUTATION_PROB) <= 1.0, (
            "The sum of the crossover and mutation probabilities must be smaller "
            "or equal to 1.0.")

        offspring = []
        for _ in range(self.lambda_):
            op_choice = random.random()
            if op_choice < EvolutionManager.CX_PROB:            # Apply crossover
                ind1, ind2 = [self.toolbox.clone(i) for i in random.sample(self.population, 2)]
                ind1, ind2 = self.toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                offspring.append(ind1)
            elif op_choice < EvolutionManager.CX_PROB + EvolutionManager.MUTATION_PROB:  # Apply mutation
                ind = self.toolbox.clone(random.choice(self.population))
                ind, = self.toolbox.mutate(ind)
                del ind.fitness.values
                offspring.append(ind)
            else:                           # Apply reproduction
                offspring.append(random.choice(self.population))

        return offspring

    def end_generation(self, stats, invalid_ind):
        
        # record only handles fitness values
        record = stats.compile(self.population) if stats is not None else {}

        # manually calc loss mean, std, med, min, max
        lt = LoggingTemplate
        losses = [indi.model_performance[lt.train_loss] for indi in self.population]

        l_mean = np.mean(losses)
        l_std = np.std(losses)
        l_med = np.median(losses)
        l_min = np.min(losses)
        l_max = np.max(losses)

        # Show status
        if EvolutionManager.VERBOSE: 
            print(f"\n\n=== GENERATION COMPLETE (0 / {EvolutionManager.GENERATIONS}) ===")
            print("avg, std, med, min, max")
            want_to_print = [record['avg'], record['std'], record['med'], record['min'], record['max']]
            want_to_print = list(map(str, list(map(lambda x: round(x, 2), want_to_print))))
            print(" ".join(want_to_print))

        # Log the generation
        if LoggingSettings.LOGGING:
            self.LogManager.log_generation_stats(self.population, len(invalid_ind), 
                                                 record['avg'], record['std'], record['med'], record['min'], record['max'],
                                                 l_mean, l_std, l_med, l_min, l_max)
    
    def _loading_bar(self, gen, max_time_per_gen, elapsed_time):
        bar_size = 60  # Total width of the progress bar
        
        # Handle empty bar case (gen = -1)
        if gen == -1:
            progress = 0
            percentage = 0.0
            time_str = "ETA: --h --m --s"
        else:
            progress = int((gen / (EvolutionManager.GENERATIONS+1)) * bar_size)
            percentage = round((gen / (EvolutionManager.GENERATIONS+1))*100, 2)
            
            # Calculate estimated time remaining
            if max_time_per_gen > 0:
                remaining_gens = EvolutionManager.GENERATIONS - gen
                remaining_time = remaining_gens * max_time_per_gen
                eta_hours, rem = divmod(remaining_time, 3600)
                eta_mins, eta_secs = divmod(rem, 60)
                time_str = f"ETA: {int(eta_hours)}h {int(eta_mins)}m {int(eta_secs)}s"
            else:
                time_str = "ETA: --h --m --s"

        remaining = bar_size - progress
        time_str = Clr(time_str, "red" if percentage <= 50 else "yellow" if percentage <= 75 else "green")

        # Calculate elapsed time
        elapsed_hours, rem = divmod(elapsed_time, 3600)
        elapsed_mins, elapsed_secs = divmod(rem, 60)
        elapsed_str = f"{int(elapsed_hours)}h {int(elapsed_mins)}m {int(elapsed_secs)}s"

        # Format generation time
        gen_hours, rem = divmod(max_time_per_gen, 3600)
        gen_mins, gen_secs = divmod(rem, 60)
        gen_time_str = f"{int(gen_hours)}h {int(gen_mins)}m {int(gen_secs)}s" if gen_hours > 0 else \
                    f"{int(gen_mins)}m {int(gen_secs)}s" if gen_mins > 0 else \
                    f"{int(gen_secs)}s"

        # Create the centered percentage bar
        percentage_text = f" {percentage}% "
        bar_center = bar_size // 2
        text_start = bar_center - len(percentage_text) // 2
        
        # Build the bar segments
        before_text = min(progress, text_start)
        after_text = progress - before_text
        bar_segments = [
            Clr(' '*before_text, bg_color="bright_green"),
            Clr(percentage_text, "black", bg_color="bright_green"),
            Clr(' '*after_text, bg_color="bright_green") if gen != -1 else Clr(' '*after_text),
            Clr(' '*remaining, bg_color="bright_white")
        ]
        # Combine all components
        print(" "*160, end="\r")  # Clear previous line
        print(f"│{''.join(str(seg) for seg in bar_segments)}│ "
            f"{time_str} │ "
            f"Elapsed: {elapsed_str} │ "
            f"Last generation duration: {gen_time_str}",
            end="\r")