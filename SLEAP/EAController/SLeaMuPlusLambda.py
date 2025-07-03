"""
Modified eaMuPlusLambda comes from the deap.algorithms.eaSimple that can be further seen here:
https://github.com/DEAP/deap/tree/master

Hornby, Greg. (2006). 
ALPS: The age-layered population structure for reducing the problem of premature convergence. 
GECCO 2006 - Genetic and Evolutionary Computation Conference. 1. 
10.1145/1143997.1144142. 
"""

import random
from Globals import EvolutionSettings, AlpsSettings, LoggingSettings

def eaMuPlusLambda(population, toolbox, cxpb, mutpb, ngen, LogManager,
                   stats=None, halloffame=None, verbose=__debug__):
    """See: DEAP/Algorithms"""

    mu = EvolutionSettings.POPULATION_SIZE
    lambda_ = mu

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    LoggingSettings.population_size = len(invalid_ind)

    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    if LoggingSettings.LOGGING:
        LogManager.log_generation_stats(0, len(invalid_ind), record['avg'], record['std'], record['med'], record['min'], record['max'], test_the_best=False)


    # Begin the generational process
    for gen in range(1, ngen + 1):
        if verbose: 
            print(f"\n\n===== NEW GEN ({gen} / {ngen})===")
            print("avg, std, med, min, max")
            want_to_print = [record['avg'], record['std'], record['med'], record['min'], record['max']]
            want_to_print = list(map(str, list(map(lambda x: round(x, 2), want_to_print))))
            print(" ".join(want_to_print))

            for thing in AlpsSettings.individuals_and_fitnesses_in_brackets.keys():
                
                print(f"Bracket {thing}:")
                for (indi, fit) in AlpsSettings.individuals_and_fitnesses_in_brackets[thing]:
                    print(f"{'Individual And Fitness:':30} {f'{indi}':30} {f'{fit}':30}")

        if int(gen) == int(ngen*(EvolutionSettings.BETA_SWITCH)):
            EvolutionSettings.alpha = 1
            EvolutionSettings.beta = 0

        # Replace Layer 0 every AGE_GAP generations
        if gen % AlpsSettings.AGE_GAP == 0:
            population_layer_0 = [ind for ind in population if ind.bracket == 0]
            num_to_replace = len(population_layer_0)
            
            new_individuals = [toolbox.individual() for _ in range(num_to_replace)]
            fitnesses = toolbox.map(toolbox.evaluate, new_individuals)
            for ind, fit in zip(new_individuals, fitnesses):
                ind.fitness.values = fit

            # Remove old bracket 0 individuals from population
            population = [ind for ind in population if ind.bracket != 0]
            population.extend(new_individuals)

            if verbose:
                print(f"🧼 Layer 0 replaced with {num_to_replace} new individuals (Generation {gen})")



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
        population[:] = toolbox.select(population + offspring, mu)
        update_brackets(population)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        if LoggingSettings.LOGGING:
            # Log the generation
            LogManager.log_generation_stats(gen, len(invalid_ind), record['avg'], record['std'], record['med'], record['min'], record['max'])

    return population

def varOr(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            crossover_child = crossover(population, toolbox)
            emptyValues(crossover_child)
            offspring.append(crossover_child)

        elif op_choice < cxpb + mutpb:  # Apply mutation

            pre_mutation_ind = random.choice(population)
            age, bracket = pre_mutation_ind.age, pre_mutation_ind.bracket

            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            emptyValues(ind)

            ind.age = age + 1
            ind.bracket = bracket

            offspring.append(ind)

        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring

def crossover(population, toolbox):

    ind1_no_clone = random.choice(population)
    ind1 = toolbox.clone( ind1_no_clone )

    other_individuals_in_same_bracket = random.choice(AlpsSettings.individuals_and_fitnesses_in_brackets[ind1.bracket])
    ind2_no_clone = other_individuals_in_same_bracket[0]
    ind2 = toolbox.clone( ind2_no_clone )

    ind1, ind2 = toolbox.mate(ind1, ind2)

    ind1.age = max( ind1_no_clone.age, ind2_no_clone.age ) + 1
    ind1.bracket = max( ind1_no_clone.bracket, ind2_no_clone.bracket )

    return ind1

def emptyValues(offspring):
    del offspring.fitness.values

    if hasattr(offspring, "raw_fitness"):
        del offspring.raw_fitness
    
    if hasattr(offspring, "uniqueness"):
        del offspring.uniqueness

def update_brackets(future_population):
    AlpsSettings.individuals_and_fitnesses_in_brackets = {}
    
    for individual in future_population:
        if individual.bracket not in AlpsSettings.individuals_and_fitnesses_in_brackets:
            AlpsSettings.individuals_and_fitnesses_in_brackets[individual.bracket] = []