"""
Modified eaSimple comes from the deap.algorithms.eaSimple that can be further seen here:
https://github.com/DEAP/deap/tree/master

The modification is that every N gerenations, the top ⌈ M% individuals ⌉ are trained with the entire data set.
From those top M individuals, the top 50% are chosen, along with a random 50% from the entire population.
We then create a new population from those individuals.
"""

import random
from math import ceil

from Globals import EvolutionSettings

def ModifiedEASimple(population, toolbox, cxpb, mutpb, ngen, LogManager, stats=None,
             halloffame=None, verbose=__debug__):
    """See: DEAP/Algorithms"""

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(invalid_ind) if stats else {}

    if EvolutionSettings.LOGGING:
        # Log the generation
        LogManager.log_generation_stats(0, len(invalid_ind), record['avg'], record['std'], record['med'], record['min'], record['max'], tournament_of_champions=False)


    # Begin the generational process
    for gen in range(1, ngen + 1):
        if verbose: 
            print(f"\n\n===== NEW GEN ({gen} / {ngen})===")
            print("avg, std, med, min, max")
            want_to_print = [record['avg'], record['std'], record['med'], record['min'], record['max']]
            want_to_print = list(map(str, list(map(lambda x: round(x, 2), want_to_print))))
            print(" ".join(want_to_print))

        if (EvolutionSettings.TOC_ON) and (gen % EvolutionSettings.TOC_GENERATIONS_BETWEEN == 0):
            population = tournament_of_champions(population, toolbox, verbose)
            tournament_of_champions_happened = True
        else:
            tournament_of_champions_happened = False

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(invalid_ind) if stats else {}

        if EvolutionSettings.LOGGING:
            # Log the generation
            LogManager.log_generation_stats(gen, len(invalid_ind), record['avg'], record['std'], record['med'], record['min'], record['max'], tournament_of_champions_happened)

    return population

def tournament_of_champions(population, toolbox, verbose=False):
    """
    Tournament of Champions: Every N generations, the top M% individuals are
    retrained on 100% of the dataset. From those, the top 50% are chosen along
    with a random 50% from the rest of the population to form a new population.
    
    Args:
        population: Current population of individuals
        toolbox: DEAP toolbox with evaluate_full_dataset method
        verbose:
    
    Returns:
        New population after tournament of champions
    """
    if verbose:
        print(f"\n=== 🔥🔥🏆 TOURNAMENT OF CHAMPIONS 🏆🔥🔥 ===")
    
    # Sort population by fitness (descending order for maximization)
    sorted_population = sorted(population, key=lambda x: x.fitness.values[0], reverse=True)
    
    # Calculate number of top individuals to retrain
    tournament_size = ceil(len(population) * EvolutionSettings.TOC_TOURNAMENT_SIZE)
    top_individuals = sorted_population[:tournament_size]
    
    if verbose:
        print(f"Retraining top {tournament_size} individuals on full dataset...")
    
    # Retrain top individuals on full dataset
    full_dataset_fitnesses = toolbox.map(toolbox.evaluate_champion, top_individuals)
    for ind, fit in zip(top_individuals, full_dataset_fitnesses):
        ind.fitness.values = fit
    
    # Sort retrained individuals by their new fitness
    retrained_sorted = sorted(top_individuals, key=lambda x: x.fitness.values[0], reverse=True)
    
    # Select top 50% of retrained individuals
    champions_count = ceil(len(retrained_sorted) * 0.5)
    champions = retrained_sorted[:champions_count]
    
    # Get remaining individuals from the rest of the population
    rest_of_population = sorted_population[tournament_size:]
    
    # Calculate how many random individuals we need to complete the population
    remaining_slots = len(population) - len(champions)
    
    # Randomly sample from the rest of the population
    if len(rest_of_population) >= remaining_slots:
        random_individuals = random.sample(rest_of_population, remaining_slots)
    else:
        # If we don't have enough individuals, sample with replacement
        random_individuals = random.choices(rest_of_population, k=remaining_slots)
    
    # Create new population
    new_population = champions + random_individuals
    
    # Clone individuals to ensure independence
    new_population = [toolbox.clone(ind) for ind in new_population]
    
    if verbose:
        print(f"New population created with {len(champions)} champions and {len(random_individuals)} random individuals")
        print(f"Champion fitness range: {champions[-1].fitness.values[0]:.4f} - {champions[0].fitness.values[0]:.4f}")
        print("=== END TOURNAMENT OF CHAMPIONS ===\n")
    
    return new_population

def varAnd(population, toolbox, cxpb, mutpb):
    """See: DEAP/Algorithms"""

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
