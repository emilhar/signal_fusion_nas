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

        if int(gen) == int(ngen*(EvolutionSettings.BETA_SWITCH)):
            EvolutionSettings.alpha = 1
            EvolutionSettings.beta = 0

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
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            emptyValues(ind)
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring


def crossover(population, toolbox):
    ind1 = random.sample(population, 1)
    other_individuals_in_same_bracket = random.choice(AlpsSettings.individuals_and_fitnesses_in_brackets[ind1.bracket])
    ind2 = other_individuals_in_same_bracket[0]

    ind1, ind2 = toolbox.mate(ind1, ind2)
    return ind1


def emptyValues(offspring):
    del offspring.fitness.values

    if hasattr(offspring, "raw_fitness"):
        del offspring.raw_fitness
    
    if hasattr(offspring, "uniqueness"):
        del offspring.uniqueness

