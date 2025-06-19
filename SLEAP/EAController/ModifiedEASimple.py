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
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}

    if EvolutionSettings.LOGGING:
        # Log the best individual in the generation
        best = LogManager.best_individual_in_generation
        LogManager.log_individual_stats(
                individual=best["individual"], 
                fitness=best["fitness"], 
                champion=best["champion"], 
                train_loss=best["train_loss"], 
                test_loss=best["test_loss"], 
                precision=best["precision"],
                recall=best["recall"], 
                f1=best["f1"], 
                accuracy=best["accuracy"]
            )
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
        record = stats.compile(population) if stats else {}

        if EvolutionSettings.LOGGING:
            # Log the best individual in the generation
            best = LogManager.best_individual_in_generation
            LogManager.log_individual_stats(
                individual=best["individual"], 
                fitness=best["fitness"], 
                champion=best["champion"], 
                train_loss=best["train_loss"], 
                test_loss=best["test_loss"], 
                precision=best["precision"],
                recall=best["recall"], 
                f1=best["f1"], 
                accuracy=best["accuracy"]
            )
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
    r"""Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
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
