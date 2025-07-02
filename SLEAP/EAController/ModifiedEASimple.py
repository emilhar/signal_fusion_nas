"""
Modified eaSimple comes from the deap.algorithms.eaSimple that can be further seen here:
https://github.com/DEAP/deap/tree/master
"""

import random
from Globals import EvolutionSettings, LoggingSettings

def ModifiedEASimple(population, toolbox, cxpb, mutpb, ngen, LogManager, stats=None,
             halloffame=None, verbose=__debug__):
    """See: DEAP/Algorithms"""

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    LoggingSettings.population_size = len(invalid_ind)
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
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

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        LoggingSettings.population_size = len(invalid_ind)
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        if LoggingSettings.LOGGING:
            # Log the generation
            LogManager.log_generation_stats(gen, len(invalid_ind), record['avg'], record['std'], record['med'], record['min'], record['max'])

    return population

def varAnd(population, toolbox, cxpb, mutpb):
    """See: DEAP/Algorithms"""
    if cxpb == 0.0 and mutpb == 0.0:
        for pop in population:
            del pop.fitness.values
        return population[:]

    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring