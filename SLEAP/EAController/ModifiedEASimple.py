"""
Modified eaSimple comes from the deap.algorithms.eaSimple that can be further seen here:
https://github.com/DEAP/deap/tree/master

Hornby, Greg. (2006). 
ALPS: The age-layered population structure for reducing the problem of premature convergence. 
GECCO 2006 - Genetic and Evolutionary Computation Conference. 1. 
10.1145/1143997.1144142. 
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
        offspring = make_next_gen(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        LoggingSettings.population_size = len(invalid_ind)
        
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

def make_next_gen(population, toolbox, cxpb, mutpb):
    """Almost the same as DEAP's varAnd.
    Now crossover application is based on age brackets"""
    if cxpb == 0.0 and mutpb == 0.0:
        for pop in population:
            del pop.fitness.values
        return population[:]

    offspring = [toolbox.clone(ind) for ind in population]

    brackets = get_brackets()

    # Apply crossover
    for offspring_member in offspring:

        # cxpb is divided by 2 due to the fact that if an individual is chosen, 
        # it will automatically select another individual in it's bracket to crossover with.
        if random.random() < cxpb/2:
            if offspring_member.bracket == 0:
                other_member = random.choice(brackets[offspring_member.bracket])
            else:
                bracket_choice = random.choice([offspring_member.bracket - 1, offspring_member.bracket])
                other_member = random.choice(brackets[bracket_choice])
            
            offspring_member, other_member, child_age = toolbox.mate(offspring_member, other_member)

            emptyValues(offspring_member)
            emptyValues(other_member)
            
            offspring_member.age = child_age
            other_member.age = child_age

    # Apply mutation
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            emptyValues(offspring[i])

    return offspring

def get_brackets(population) -> dict:
    brackets = {}

    for individual in population:
        if individual.bracket not in brackets:
            brackets[individual.bracket] = []
        
        brackets[individual.bracket].append(individual)

    return brackets

def emptyValues(offspring):
    del offspring.fitness.values

    if hasattr(offspring, "raw_fitness"):
        del offspring.raw_fitness
    
    if hasattr(offspring, "uniqueness"):
        del offspring.uniqueness

