from typing import Callable, List, Tuple, Dict
import random

Individual = Tuple[float]
Population = List[Individual]
PopulationFitness = Dict[Individual, float]


def genetic_algorithm(
        terminate_fn: Callable[[Population, Population], bool],
        initialize_fn: Callable[[], Population],
        fitness_fn: Callable[[Individual], float],
        select_fn: Callable[[Population, PopulationFitness], Population],
        crossover_fn: Callable[[Population], Population],
        mutation_fn: Callable[[Individual], Individual],
        replace_fn: Callable[[Population, Population, PopulationFitness], Population],
) -> Population:
    population = initialize_fn()
    population_size = len(population)
    fitness_by_individual = {individual: fitness_fn(individual) for individual in population}

    while True:
        all_children = []
        while len(all_children) != population_size:
            parents = select_fn(population, fitness_by_individual)
            children = crossover_fn(parents)
            for child in children:
                child = mutation_fn(child)
                fitness_by_individual[child] = fitness_fn(child)
                all_children.append(child)

        new_population = replace_fn(population, all_children, fitness_by_individual)
        fitness_by_individual = {individual: fitness_by_individual[individual] for individual in new_population}

        old_population = population
        population = new_population

        if terminate_fn(old_population, population):
            break

    return population


def differential_evolution(
        terminate_fn: Callable[[Population, Population], bool],
        initialize_fn: Callable[[], Population],
        fitness_fn: Callable[[Individual], float],
        fitness_is_better_fn: Callable[[float, float], bool],
        crossover_rate: float, differential_weight: float
) -> Population:
    population = initialize_fn()
    fitness_by_individual = {individual: fitness_fn(individual) for individual in population}

    while True:
        new_population = []
        for i, individual in enumerate(population):
            other_individuals = list(range(len(population)))
            other_individuals.remove(i)
            a, b, c = random.sample(other_individuals, 3)

            new_individual = list(individual)
            j_rand = random.randrange(len(individual))
            for j in range(len(individual)):
                mutated = a[j] + differential_weight * (b[j] - c[c])
                trial = mutated if random.uniform(0, 1) <= crossover_rate or j == j_rand else individual[j]
                new_individual[j] = trial
            new_individual = type(individual)(new_individual)
            fitness_by_individual[new_individual] = fitness_fn(new_individual)

            if fitness_is_better_fn(fitness_by_individual[new_individual], fitness_by_individual[individual]):
                new_population.append(new_individual)
            else:
                new_population.append(individual)

        fitness_by_individual = {individual: fitness_by_individual[individual] for individual in new_population}

        old_population = population
        population = new_population

        if terminate_fn(old_population, population):
            break

    return population


def particle_swarm_optimization(

) -> Population:
