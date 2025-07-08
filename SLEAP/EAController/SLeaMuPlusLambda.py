"""
Modified eaMuPlusLambda comes from the deap.algorithms.eaSimple that can be further seen here:
https://github.com/DEAP/deap/tree/master

Hornby, Greg. (2006). 
ALPS: The age-layered population structure for reducing the problem of premature convergence. 
GECCO 2006 - Genetic and Evolutionary Computation Conference. 1. 
10.1145/1143997.1144142. 
"""



""" Taka út max age í síðasta layer """

import random
from Globals import EvolutionSettings, AlpsSettings, LoggingSettings, FitnessFunctions, SLEAP_Exception

def eaMuPlusLambda(population, toolbox, cxpb, mutpb, ngen, LogManager,
                   stats=None, halloffame=None, verbose=__debug__):
    """See: DEAP/Algorithms
    mu: The number of individuals to select for the next generation.
    lambda: The number of children to produce at each generation."""

    mu = EvolutionSettings.POPULATION_SIZE_PER_LAYER
    lambda_ = mu // 2

    __dev_check(function=eaMuPlusLambda.__name__, line=23)

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

    update_individuals_and_fitnesses_in_layer(population, 0)

    if LoggingSettings.LOGGING:
        for individual in population:
            LogManager.check_for_best_in_gen(individual)
            
        LogManager.log_generation_stats(len(invalid_ind), record['avg'], record['std'], record['med'], record['min'], record['max'])

    __dev_check(function=eaMuPlusLambda.__name__, line=47)
    # Begin the generational process
    for gen in range(1, ngen + 1):
        __dev_check(function=eaMuPlusLambda.__name__, line=50)

        LoggingSettings.population_size =       len(population)
        LoggingSettings.current_generation_id = gen
        LoggingSettings.current_individual_id = 0

        if verbose: 
            print_individual_dict()
            print(f"\n\n===== NEW GEN ({gen} / {ngen})===")
            print("avg, std, med, min, max")
            want_to_print = [record['avg'], record['std'], record['med'], record['min'], record['max']]
            want_to_print = list(map(str, list(map(lambda x: round(x, 2), want_to_print))))
            print(" ".join(want_to_print))

        __dev_check(function=eaMuPlusLambda.__name__, line=64)
        for layer in sorted(AlpsSettings.individuals_and_fitnesses_in_layers.keys()):
            if verbose: print(f"\nWorking on layer {layer}")

            layer_population = [toolbox.clone(indi) for indi in population if indi.layer == layer]

            # Vary the population
            offspring, genetic_material_used = varOr(layer_population, toolbox, lambda_, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            __dev_check(function=eaMuPlusLambda.__name__, line=83)

            # Select the next generation population
            combined_population = layer_population + offspring
            new_chosen_population =  toolbox.select(combined_population, mu)
            __dev_check(function=eaMuPlusLambda.__name__, line=88)
            
            individuals_to_age = []
    
            for individual in new_chosen_population:
                ind_id = id(individual)
                if ind_id not in genetic_material_used.keys():
                    continue

                for parent in genetic_material_used[ind_id]:
                    if parent not in individuals_to_age:
                        individuals_to_age.append(parent)

            for individual in individuals_to_age:
                individual.age += 1

            population = [indi for indi in population if indi.layer != layer]

            for new_individual in new_chosen_population:
                if individual not in population:
                    population.append(new_individual)

            __dev_check(function=eaMuPlusLambda.__name__, line=106)
            update_individuals_and_fitnesses_in_layer(population, layer)
            __dev_check(function=eaMuPlusLambda.__name__, line=108)
        
        if gen in AlpsSettings.LAYER_CREATION_THRESHOLDS:
            create_new_layer(population, toolbox)
        __dev_check(function=eaMuPlusLambda.__name__, line=112)
        manage_layer_transitions(population)
        __dev_check(function=eaMuPlusLambda.__name__, line=114)
        population = check_for_empty_layers(population, toolbox)
        __dev_check(function=eaMuPlusLambda.__name__, line=116)

        # Replace Layer 0 every AGE_GAP generations
        if (gen % AlpsSettings.AGE_GAP == 0):

            if verbose: print("\n\n## Replacing Layer 0 ##\n")
            if verbose: print(f"New additions:", mu)

            new_individuals = [toolbox.individual() for _ in range(mu)]
            fitnesses = toolbox.map(toolbox.evaluate, new_individuals)
            for ind, fit in zip(new_individuals, fitnesses):
                ind.fitness.values = fit

            # Remove old layer 0 individuals from population
            population = [ind for ind in population if ind.layer != 0]
            population.extend(new_individuals)
            update_individuals_and_fitnesses_in_layer(population, 0)
        __dev_check(function=eaMuPlusLambda.__name__, line=133)
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        if LoggingSettings.LOGGING:
            # Log the generation
            LogManager.log_generation_stats(len(invalid_ind), record['avg'], record['std'], record['med'], record['min'], record['max'])

    return population

def update_individuals_and_fitnesses_in_layer(population, layer_to_update):
    """Update a layer's population by filtering the global population"""

    AlpsSettings.individuals_and_fitnesses_in_layers[layer_to_update] = [
        (ind, ind.fitness.values[0])
        for ind in population 
        if ind.layer == layer_to_update
    ]

def varOr(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    genetic_material_used = {}

    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            crossover_child, parent_1, parent_2 = crossover(population, toolbox)

            offspring.append(crossover_child)
            genetic_material_used[id(crossover_child)] = [parent_1, parent_2]

        elif op_choice < cxpb + mutpb:  # Apply mutation

            mutant, ind_pre_mutation = mutate(population, toolbox)

            offspring.append(mutant)
            genetic_material_used[id(mutant)] = [ind_pre_mutation]

        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    __dev_check(function=varOr.__name__)
    return offspring, genetic_material_used

def print_individual_dict():
    # Print layer information in a table format with dynamic column widths
    for layer in sorted(AlpsSettings.individuals_and_fitnesses_in_layers.keys()):
        layer_data = AlpsSettings.individuals_and_fitnesses_in_layers[layer]
        if not layer_data:
            continue
        
        # Calculate max lengths for each column
        max_ind_len = max(len(str(indi)) for (indi, _) in layer_data)
        max_age_len = max(len(f"{indi.age}/{AlpsSettings.MAX_AGE_IN_LAYERS[indi.layer]}") 
                        for (indi, _) in layer_data)
        max_fit_len = max(len(f"{fit:.4f}") for (_, fit) in layer_data)
        
        # Add some padding (minimum 10 for ind, 8 for others)
        ind_width = max(10, max_ind_len + 2)
        age_width = max(8, max_age_len + 2)
        fit_width = max(8, max_fit_len + 2)
        
        # Print table header
        header = f"\nLayer {layer}:"
        separator = "-" * (ind_width + age_width + fit_width + 4)
        print(header)
        print(separator)
        print(f"{'Individual':<{ind_width}} {'Age':<{age_width}} {'Fitness':<{fit_width}}")
        print(separator)
        
        # Print each row
        for (indi, fit) in layer_data:
            age_str = f"{indi.age}/{AlpsSettings.MAX_AGE_IN_LAYERS[indi.layer]}"
            fit_str = f"{fit:.4f}"
            print(f"{str(indi):<{ind_width}} {age_str:<{age_width}} {fit_str:<{fit_width}}")
        
        print(separator)

def crossover(population, toolbox):

    ind1 = random.choice(population)
    ind1_clone = toolbox.clone( ind1 )

    if ind1.layer != 0:
        layer_choice = random.choice([ind1.layer, ind1.layer-1])
    else:
        layer_choice = ind1.layer

    other_individuals_in_same_layer = random.choice(AlpsSettings.individuals_and_fitnesses_in_layers[layer_choice])
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

def create_new_layer(population, toolbox):
    print("Creating new layer")
    # Get the current maximum layer
    max_layer = max(AlpsSettings.individuals_and_fitnesses_in_layers.keys())
    new_layer = max_layer + 1
    
    # Create new layer entry
    AlpsSettings.individuals_and_fitnesses_in_layers[new_layer] = []
    
    # Get parent individuals from highest existing layer
    parent_population = [ind for ind, _ in AlpsSettings.individuals_and_fitnesses_in_layers[max_layer]]
    
    # Generate offspring using variation operators
    offspring, _ = varOr(parent_population, toolbox, 
                     EvolutionSettings.POPULATION_SIZE_PER_LAYER, 
                     EvolutionSettings.CX_PROB, 
                     EvolutionSettings.MUTATION_PROB)
    # Evaluate new offspring
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    if invalid_ind:
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    
    # Assign new layer to offspring
    for ind in offspring:
        ind.layer = new_layer

    population.extend(offspring)
    
    # Update layer registry
    update_individuals_and_fitnesses_in_layer(population, new_layer)



    __dev_check(function=create_new_layer.__name__, line=293)

def manage_layer_transitions(population):
    """Controls layer switching for all layers after population has been settled"""

    for i in range(len(AlpsSettings.individuals_and_fitnesses_in_layers)):

        layer_population = [toolbox.clone(individual) for individual in population if individual.layer==i]

        failures = []
        for individual in layer_population:
            if individual.age > AlpsSettings.MAX_AGE_IN_LAYERS[individual.layer]:
                # Now it's time to see if they move up a layer or fail to do so.
                successful = attempt_layer_switch(individual, population)

                if not successful:
                    failures.append(individual)

        for failure in failures:
            population.remove(failure)
            update_individuals_and_fitnesses_in_layer(population, failure.layer)
        
    __dev_check(function=manage_layer_transitions.__name__)

def check_for_empty_layers(population, toolbox):

    # Reset the dictionary based on the current population
    number_of_current_layers = len(AlpsSettings.individuals_and_fitnesses_in_layers)
    
    for i in range(number_of_current_layers):
        AlpsSettings.individuals_and_fitnesses_in_layers[i] = [
            (ind, ind.fitness.values[0])
            for ind in population 
            if ind.layer == i
        ]


    for layer in AlpsSettings.individuals_and_fitnesses_in_layers.keys():
        layer_population = [indi for indi in population if indi.layer == layer]

        if len(layer_population) < 2:
            print("Missing individuals in layer", layer, ". Refilling")
            new_individuals = [toolbox.individual() for _ in range( 2 - len(layer_population) )]
            fitnesses = toolbox.map(toolbox.evaluate, new_individuals)
            for ind, fit in zip(new_individuals, fitnesses):
                ind.fitness.values = fit

            population.extend(new_individuals)

    __dev_check(function=check_for_empty_layers.__name__, line=342)

    return population

def attempt_layer_switch(individual, population):
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
    
    __dev_check(function=attempt_layer_switch.__name__, line=360)
    if replace:
        print(f"Replacing {worst_individual_in_above_layer[0]} with {individual}")

        population.remove(worst_individual_in_above_layer[0])
        individual.layer += 1
        update_individuals_and_fitnesses_in_layer(population, individual.layer - 1)
        update_individuals_and_fitnesses_in_layer(population, individual.layer)
        
        __dev_check(function=attempt_layer_switch.__name__, line=369)
        return True
    
    # If the individual didn't get in, it will not be a part of the population anymore.
    return False

def __dev_check(**kwargs):
    for item in AlpsSettings.individuals_and_fitnesses_in_layers:
        for individual,_ in AlpsSettings.individuals_and_fitnesses_in_layers[item]:
            if individual.layer > item:
                raise SLEAP_Exception(**kwargs)
            
    people = []

    for item in AlpsSettings.individuals_and_fitnesses_in_layers:
        for myTuple in AlpsSettings.individuals_and_fitnesses_in_layers[item]:
            if myTuple not in people:
                people.append(myTuple)
            else:
                print("Second error")
                raise SLEAP_Exception(**kwargs)
