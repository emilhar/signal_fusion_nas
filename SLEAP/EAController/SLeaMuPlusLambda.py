"""
Modified eaMuPlusLambda comes from the deap.algorithms.eaSimple that can be further seen here:
https://github.com/DEAP/deap/tree/master

Hornby, Greg. (2006). 
ALPS: The age-layered population structure for reducing the problem of premature convergence. 
GECCO 2006 - Genetic and Evolutionary Computation Conference. 1. 
10.1145/1143997.1144142. 
"""

import random
from Globals import EvolutionSettings, AlpsSettings, LoggingSettings, FitnessFunctions

def eaMuPlusLambda(population, toolbox, cxpb, mutpb, ngen, LogManager,
                   stats=None, halloffame=None, verbose=__debug__):
    """See: DEAP/Algorithms"""

    mu = EvolutionSettings.POPULATION_SIZE
    lambda_ = mu // 2

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    LoggingSettings.population_size = len(invalid_ind)
    LoggingSettings.current_generation_id = 0

    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}

    update_layers(population)

    if LoggingSettings.LOGGING:
        for individual in population:
            LogManager.check_for_best_in_gen(individual)
            
        LogManager.log_generation_stats(len(invalid_ind), record['avg'], record['std'], record['med'], record['min'], record['max'])


    # Begin the generational process
    for gen in range(1, ngen + 1):
        LoggingSettings.current_individual_id = 0
        for guy in population:
            print(guy)
        if verbose: 
            print(f"\n\n===== NEW GEN ({gen} / {ngen})===")
            print("avg, std, med, min, max")
            want_to_print = [record['avg'], record['std'], record['med'], record['min'], record['max']]
            want_to_print = list(map(str, list(map(lambda x: round(x, 2), want_to_print))))
            print(" ".join(want_to_print))

            for thing in AlpsSettings.individuals_and_fitnesses_in_layers.keys():
                
                print(f"Layer {thing}:")
                for (indi, fit) in AlpsSettings.individuals_and_fitnesses_in_layers[thing]:
                    print(f"Individual And Fitness: {f'{indi}':30} {f'{indi.age}':30} {f'{fit}':30}")

        if int(gen) == int(ngen*(EvolutionSettings.BETA_SWITCH)):
            EvolutionSettings.alpha = 1
            EvolutionSettings.beta = 0

        LoggingSettings.current_generation_id = gen

        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        LoggingSettings.population_size = len(invalid_ind)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        combined_population = population + offspring
        population[:] = toolbox.select(combined_population, mu)
        update_layers(population)

        # Replace Layer 0 every AGE_GAP generations
        if (gen > AlpsSettings.AGE_GAP) and (gen % AlpsSettings.AGE_GAP == 1):

            if verbose: print("\n\n## Replacing Layer 0 ##\n")

            if verbose: print(f"New additions:", lambda_)
            new_individuals = [toolbox.individual() for _ in range(lambda_)]
            fitnesses = toolbox.map(toolbox.evaluate, new_individuals)
            for ind, fit in zip(new_individuals, fitnesses):
                ind.fitness.values = fit

            # Remove old layer 0 individuals from population
            population = [ind for ind in population if ind.layer != 0]
            population.extend(new_individuals)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        if LoggingSettings.LOGGING:
            # Log the generation
            LogManager.log_generation_stats(len(invalid_ind), record['avg'], record['std'], record['med'], record['min'], record['max'])

    return population

def varOr(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    genetic_material_used = []

    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            crossover_child, parent_1, parent_2 = crossover(population, toolbox)

            offspring.append(crossover_child)
            if parent_1 not in genetic_material_used: genetic_material_used.append(parent_1)
            if parent_2 not in genetic_material_used: genetic_material_used.append(parent_2)

        elif op_choice < cxpb + mutpb:  # Apply mutation

            mutant, ind_pre_mutation = mutate(population, toolbox)

            offspring.append(mutant)
            if ind_pre_mutation not in genetic_material_used: genetic_material_used.append(ind_pre_mutation)

        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    for parent in genetic_material_used:
        parent.age += 1

    return offspring

def crossover(population, toolbox):

    ind1 = random.choice(population)
    ind1_clone = toolbox.clone( ind1 )

    other_individuals_in_same_layer = random.choice(AlpsSettings.individuals_and_fitnesses_in_layers[ind1.layer])
    ind2 = other_individuals_in_same_layer[0]
    ind2_clone = toolbox.clone( ind2 )

    ind1_clone, ind2_clone = toolbox.mate(ind1_clone, ind2_clone)

    ind1_clone.age = max( ind1.age, ind2.age ) + 1
    ind1_clone.layer = max( ind1.layer, ind2.layer )

    emptyValues(ind1_clone)

    return ind1_clone, ind1, ind2

def mutate(population, toolbox):
    pre_mutation_ind = random.choice(population)

    ind = toolbox.clone( pre_mutation_ind )
    ind, = toolbox.mutate(ind)
    emptyValues(ind)

    ind.age = pre_mutation_ind.age + 1
    ind.layer = pre_mutation_ind.layer

    return ind, pre_mutation_ind

def emptyValues(offspring):
    del offspring.fitness.values

    if hasattr(offspring, "raw_fitness"):
        del offspring.raw_fitness
    
    if hasattr(offspring, "uniqueness"):
        del offspring.uniqueness

def update_layers(population):
    # Clear layers and refill them based on the current population
    AlpsSettings.individuals_and_fitnesses_in_layers = {}

    for individual in population:
        if individual.layer not in AlpsSettings.individuals_and_fitnesses_in_layers:
            AlpsSettings.individuals_and_fitnesses_in_layers[individual.layer] = []

        AlpsSettings.individuals_and_fitnesses_in_layers[individual.layer].append( (individual, individual.fitness.values[0]) )

    failures = []
    # Now that the layers are correct,
    # we must see if individuals that have aged out of their layers can move to the next one.
    for individual in population:
        if individual.age > AlpsSettings.MAX_AGE_IN_LAYERS[individual.layer]:
            # Now it's time to see if they move up a layer or fail to do so.
            successful = attempt_layer_switch(individual)

            if not successful:
                failures.append(individual)

    for failure in failures:
        population.remove(failure)
        AlpsSettings.individuals_and_fitnesses_in_layers[failure.layer].remove( (failure, failure.fitness.values[0]) )

def attempt_layer_switch(individual):
        
        # If a new layer JUST opened, we're allowed in
        if LoggingSettings.current_generation_id == (AlpsSettings.MAX_AGE_IN_LAYERS[individual.layer] + 1):

            if individual.layer + 1 not in AlpsSettings.individuals_and_fitnesses_in_layers:
                AlpsSettings.individuals_and_fitnesses_in_layers[individual.layer + 1] = []

            AlpsSettings.individuals_and_fitnesses_in_layers[individual.layer].remove( (individual, individual.fitness.values[0]) )
            AlpsSettings.individuals_and_fitnesses_in_layers[individual.layer + 1].append( (individual, individual.fitness.values[0]) )

            individual.layer += 1

            return True
        
        # If the layer is not new, the individual must be better than the worst person in the above layer
        else:
            individuals_in_above_layer = AlpsSettings.individuals_and_fitnesses_in_layers[individual.layer + 1]
            replace = False

            if FitnessFunctions.MINIMIZE_FITNESS:
                worst_individual_in_above_layer = max(individuals_in_above_layer, key=lambda x: x[1])
                if worst_individual_in_above_layer[1] > individual.fitness.values[0]:
                    replace = True

            else:
                worst_individual_in_above_layer = min(individuals_in_above_layer, key=lambda x: x[1])
                if worst_individual_in_above_layer[1] < individual.fitness.values[0]:
                    replace = True
                    
                    
            if replace:
                print("Replacing")
                print(worst_individual_in_above_layer)
                print("with")
                print(individual)

                AlpsSettings.individuals_and_fitnesses_in_layers[individual.layer].remove( (individual, individual.fitness.values[0]) )
                AlpsSettings.individuals_and_fitnesses_in_layers[individual.layer + 1].remove( worst_individual_in_above_layer )
                AlpsSettings.individuals_and_fitnesses_in_layers[individual.layer + 1].append( (individual, individual.fitness.values[0]) )
                
                individual.layer += 1

                return True
            
        # If the layer wasn't new, and the individual didn't get in, it will not be a part of the population anymore.
        return False