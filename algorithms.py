from typing import Callable, List, TypeVar, Dict, Hashable, Iterable, Sized

Individual = TypeVar('T', Hashable, Iterable, Sized)
Population = List[Individual]
PopulationFitness = Dict[Individual, float]


def genetic_algorithm(
        num_generations,
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

    for generation in range(num_generations):
        new_population = []
        while len(new_population) != population_size:
            parents = select_fn(population, fitness_by_individual)
            children = crossover_fn(parents)
            for child in children:
                child = mutation_fn(child)
                fitness_by_individual[child] = fitness_fn(child)
                new_population.append(child)

        population = replace_fn(population, new_population, fitness_by_individual)
        fitness_by_individual = {individual: fitness_by_individual[individual] for individual in population}
        print(generation, fitness_by_individual)

    return population
