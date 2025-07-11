"""
Modified eaMuPlusLambda comes from the deap.algorithms.eaSimple that can be further seen here:
https://github.com/DEAP/deap/tree/master

Hornby, Greg. (2006). 
ALPS: The age-layered population structure for reducing the problem of premature convergence. 
GECCO 2006 - Genetic and Evolutionary Computation Conference. 1. 
10.1145/1143997.1144142. 
"""

import random
from Globals import EvolutionManager, AlpsManager, LoggingSettings, Clr

class SLeaMuPlusLambda:
    def __init__(self, population, toolbox, mu, lambda_, halloffame, LogManager):
        self.population = population
        self.toolbox = toolbox
        self.mu = mu
        self.lambda_ = lambda_
        self.halloffame = halloffame
        self.LogManager = LogManager
        
    def main(self, stats):
        """See: DEAP/Algorithms
            mu: The number of individuals to select for the next generation.
            lambda: The number of children to produce at each generation."""
        
        if EvolutionManager.VERBOSE: print("=== FIRST GENERATION ===")

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

        # Begin the generational process
        for gen in range(1, EvolutionManager.GENERATIONS + 1):

            if EvolutionManager.VERBOSE: print("\n=== NEW GENERATION ===")

            # Save info to LoggingSettings to help verbosity 
            LoggingSettings.population_size = len(self.population)
            LoggingSettings.current_generation_id = gen
            LoggingSettings.current_individual_id = 0

            # Evolve each layer
            all_layers = set(individual.layer for individual in self.population)
            for layer in range(len(all_layers)):
                if any([person for person in self.population if person.layer == layer]):
                    self.isolated_evolution(layer_to_evolve= layer)

            # Check if you need to create a new layer
            if gen in AlpsManager.LAYER_CREATION_THRESHOLDS:
                self.create_new_layer()

            # Move individuals who have aged out of their layer
            self.manage_layer_tranitions()

            # Replace layer 0
            if gen % AlpsManager.AGE_GAP == 0:
                self.replace_layer_zero()

            self.end_generation(stats, invalid_ind)

        return self.population

    def end_generation(self, stats, invalid_ind):

        record = stats.compile(self.population) if stats is not None else {}

        # Delete duplicates
        self.delete_duplicates()

        # Show status
        if EvolutionManager.VERBOSE: 
            print(f"\n\n=== GENERATION COMPLETE (0 / {EvolutionManager.GENERATIONS}) ===")
            self.print_layered_population()
            print("avg, std, med, min, max")
            want_to_print = [record['avg'], record['std'], record['med'], record['min'], record['max']]
            want_to_print = list(map(str, list(map(lambda x: round(x, 2), want_to_print))))
            print(" ".join(want_to_print))

        # Log the generation
        if LoggingSettings.LOGGING:
            for individual in self.population:
                self.LogManager.check_for_best_in_gen(individual)
                
            self.LogManager.log_generation_stats(self.population, len(invalid_ind), record['avg'], record['std'], record['med'], record['min'], record['max'])

    def isolated_evolution(self, layer_to_evolve):
        """Runs a single evolution for a single layer"""
        
        # Get every individual that is a part of that layer
        layer_population = [indi for indi in self.population if indi.layer == layer_to_evolve]
        
        if layer_to_evolve != 0:
            previous_layer_population = [indi for indi in self.population if indi.layer == layer_to_evolve - 1]
        else:
            previous_layer_population = []

        # Vary the population
        offspring, genetic_material_used = self.varOr(self.lambda_, layer_population, previous_layer_population)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if self.halloffame is not None:
            self.halloffame.update(offspring)

        # Select the next generation population
        layer_population_with_offspring = layer_population + offspring
        going_into_population = self.toolbox.select(layer_population_with_offspring, self.mu)

        # Age parents of selected individuals
        self.age_parents(going_into_population, genetic_material_used)

        # Update the population with the new layer_population
        self.population = [individual for individual in self.population if individual.layer != layer_to_evolve]
        self.population.extend( going_into_population )

    def varOr(self, lambda_, layer_population, previous_layer_population):
        """Does a crossover / mutation / reproduction lambda times for the chosen layer_population"""

        assert (EvolutionManager.CX_PROB + EvolutionManager.MUTATION_PROB) <= 1.0, (
            "The sum of the crossover and mutation probabilities must be smaller "
            "or equal to 1.0.")

        offspring = []
        genetic_material_used = {}

        for _ in range(lambda_):
            op_choice = random.random()
            if op_choice < EvolutionManager.CX_PROB:            # Apply crossover
                parents = random.sample(layer_population + previous_layer_population, 2)
                ind1, ind2 = [self.toolbox.clone(i) for i in parents]
                ind1, ind2 = self.toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                ind1.age = max(ind1.age, ind2.age) + 1
                ind1.layer = max(ind1.layer, ind2.layer)

                genetic_material_used[id(ind1)] = parents
                offspring.append(ind1)

            elif op_choice < EvolutionManager.CX_PROB + EvolutionManager.MUTATION_PROB:  # Apply mutation
                pre_mutation = random.choice(layer_population)
                ind = self.toolbox.clone(pre_mutation)
                ind, = self.toolbox.mutate(ind)
                del ind.fitness.values
                ind.age = pre_mutation.age + 1
                ind.layer = pre_mutation.layer

                genetic_material_used[id(ind)] = [pre_mutation]
                offspring.append(ind)

            else:                           # Apply reproduction
                offspring.append(ind := self.toolbox.clone( original := random.choice(layer_population)))
                del ind.fitness.values
                ind.age = original.age
                ind.layer = original.layer

        return offspring, genetic_material_used
    
    def age_parents(self, new_population, child_to_parent_dict):
        """Takes in a list of selected individuals and a dictionary containing {'child':[parents]} pairs"""

        marked = []

        for possible_child in new_population:
            if id(possible_child) not in child_to_parent_dict:
                continue

            for parent in child_to_parent_dict[id(possible_child)]:
                if parent not in marked:
                    parent.age += 1
                    marked.append(parent)
                
    def create_new_layer(self):
        # Get the current maximum layer
        current_highest_layer = max([individual.layer for individual in self.population])
        new_layer = current_highest_layer + 1

        # Get all individuals from the highest layer
        highest_layer_population = [
            individual for individual in self.population 
            if individual.layer == current_highest_layer
        ]

        # Create offspring to fill new layer with
        offspring, _ = self.varOr(self.mu, highest_layer_population, [])

        # Evaluate new offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        
        # Assign new layer to offspring
        for ind in offspring:
            ind.layer = new_layer

        # Add the new layer population to the population
        self.population.extend(offspring)

    def manage_layer_tranitions(self):
        """Controls layer switching for all layers after population has been settled"""

        for individual in self.population:
            if individual.layer == 6:
                continue
            if individual.age > AlpsManager.MAX_AGE_IN_LAYERS[individual.layer]:
                individual.layer += 1

    def replace_layer_zero(self):
        if EvolutionManager.VERBOSE: print("\n\n## Replacing Layer 0 ##\n")
        if EvolutionManager.VERBOSE: print(f"New additions:", AlpsManager.TRAINING_Manager_FOR_LAYERS[0]["mu"])

        new_individuals = [self.toolbox.individual() for _ in range( AlpsManager.TRAINING_Manager_FOR_LAYERS[0]["mu"] )]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, new_individuals)
        for ind, fit in zip(new_individuals, fitnesses):
            ind.fitness.values = fit

        # Remove old layer 0 individuals from population
        self.population = [ind for ind in self.population if ind.layer != 0]
        self.population.extend(new_individuals)

    def print_layered_population(self):
        """Prints all individuals grouped by layer in a justified table format"""
        # Group individuals by layer
        layers = {}
        for ind in self.population:
            if ind.layer not in layers:
                layers[ind.layer] = []
            layers[ind.layer].append(ind)
        
        # Sort layers
        sorted_layers = sorted(layers.items())
        
        # Print header
        print("\n" + "="*80)
        print(f"{Clr('Population by Layer', 'green')}")
        print("="*80)
        
        for layer, individuals in sorted_layers:
            # Print layer header
            print(f"\n{Clr(f'Layer {layer}', 'blue')} (size: {len(individuals)})")
            print("-"*80)
            print(f"{'Individual':<50} | {'Age':<5} | {'Fitness':<20}")
            print("-"*80)
            
            # Print individuals in this layer
            for _, ind in enumerate(individuals, 1):
                # Format individual string
                ind_str = str(ind)
                if len(ind_str) > 47:  # Truncate to fit column
                    ind_str = ind_str[:44] + "..."
                
                # Format fitness (handle cases where fitness might not be set)
                fitness_str = str(ind.fitness.values[0]) if ind.fitness.valid else "Not evaluated"
                
                # Print row
                print(f"{ind_str:<50} | {f'{ind.age}/{AlpsManager.MAX_AGE_IN_LAYERS[ind.layer]}':<5} | {fitness_str:<20}")
        
        print("="*80 + "\n")

    def delete_duplicates(self):
        seen = {}
        new_population = []

        for individual in self.population:
            key = str(individual)
            
            if key not in seen:
                new_population.append(individual)
                seen[key] = True

            else:
                # Mutate to create a new unique individual
                mutated, = self.toolbox.mutate(individual)
                new_key = str(mutated)
                
                if new_key not in seen:
                    mutated.fitness.values = self.toolbox.evaluate(mutated)
                    new_population.append(mutated)
                    seen[new_key] = True

        self.population = new_population