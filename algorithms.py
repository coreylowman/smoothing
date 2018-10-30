from typing import Callable, List, Tuple, Dict
import random
from math import log
import numpy

Individual = Tuple[float]
Population = List[Individual]
PopulationFitness = Dict[Individual, float]


def genetic_algorithm(
        terminate_fn: Callable[[Population, PopulationFitness], bool],
        observe_fn: Callable[[Population, PopulationFitness], None],
        initialize_fn: Callable[[], Population],
        fitness_fn: Callable[[Individual], float],
        select_fn: Callable[[Population, PopulationFitness], Population],
        crossover_fn: Callable[[Population], Population],
        mutation_fn: Callable[[Individual], Individual],
        replace_fn: Callable[[Population, Population, PopulationFitness], Population],
) -> Individual:
    population = initialize_fn()
    population_size = len(population)
    fitness_by_individual = {individual: fitness_fn(individual) for individual in population}

    while True:
        all_children = []
        while len(all_children) <= population_size:
            parents = select_fn(population, fitness_by_individual)
            children = crossover_fn(parents)
            for child in children:
                child = mutation_fn(child)
                fitness_by_individual[child] = fitness_fn(child)
                all_children.append(child)

        population = replace_fn(population, all_children, fitness_by_individual)
        fitness_by_individual = {individual: fitness_by_individual[individual] for individual in population}

        observe_fn(population, fitness_by_individual)
        if terminate_fn(population, fitness_by_individual):
            break

    return min(population, key=fitness_by_individual.get)


def differential_evolution(
        terminate_fn: Callable[[Population, PopulationFitness], bool],
        observe_fn: Callable[[Population, PopulationFitness], None],
        initialize_fn: Callable[[], Population],
        fitness_fn: Callable[[Individual], float],
        crossover_rate: float, differential_weight: float,
) -> Individual:
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

            if fitness_by_individual[new_individual] < fitness_by_individual[individual]:
                new_population.append(new_individual)
            else:
                new_population.append(individual)

        fitness_by_individual = {individual: fitness_by_individual[individual] for individual in new_population}
        population = new_population

        observe_fn(population, fitness_by_individual)
        if terminate_fn(population, fitness_by_individual):
            break

    return population


def particle_swarm_optimization(
        terminate_fn: Callable[[Population, PopulationFitness], bool],
        observe_fn: Callable[[Population, PopulationFitness], None],
        fitness_fn: Callable[[Individual], float],
        mins: List[float], maxs: List[float],
        population_size=40, c1=0.5 + log(2), c2=0.5 + log(2), omega=1 / (2 * log(2)),
) -> Individual:
    """
    Standard Particle Swarm Optimisation 2011 at CEC-2013: A baseline for future PSO improvements
    """

    def hypersphere(center, radius):
        offset = numpy.random.rand(len(center))
        return center + (radius * random.uniform(0, 1)) * offset / numpy.linalg.norm(offset)

    positions = []
    velocities = []
    fitnesses = []
    personal_bests = []
    personal_bests_fitnesses = []

    D = len(mins)

    for i in range(population_size):
        position = numpy.zeros(D)
        velocity = numpy.zeros(D)
        for d in range(D):
            position[d] = random.uniform(mins[d], maxs[d])
            velocity[d] = (random.uniform(mins[d], maxs[d]) - position[d]) / 2.0

        positions.append(position)
        velocities.append(velocity)
        fitnesses.append(fitness_fn(position))
        personal_bests.append(position)
        personal_bests_fitnesses.append(fitnesses[-1])

    local_best_i = min(range(population_size), key=lambda i: fitnesses[i])
    local_best = positions[local_best_i].copy()
    local_best_fitness = fitnesses[local_best_i]

    while True:
        for i in range(population_size):
            position = positions[i]
            velocity = velocities[i]
            personal_best = personal_bests[i]

            u1 = numpy.random.rand(D)
            u2 = numpy.random.rand(D)

            p = position + c1 * u1 * (personal_best - position)
            l = position + c2 * u2 * (local_best - position)
            g = (position + p + l) / 3

            h = hypersphere(g, numpy.linalg.norm(g - position))

            velocity *= omega
            velocity += h - position

            for d in range(D):
                if velocity[d] < -maxs[d]:
                    velocity[d] = -maxs[d]
                elif velocity[d] > maxs[d]:
                    velocity[d] = maxs[d]

            position += velocity

            for d in range(D):
                if position[d] < mins[d]:
                    position[d] = mins[d]
                    velocity[d] = 0
                elif position[d] > maxs[d]:
                    position[d] = maxs[d]
                    velocity[d] = 0

            fitnesses[i] = fitness_fn(position)

            if fitnesses[i] < personal_bests_fitnesses[i]:
                personal_bests[i] = position.copy()
                personal_bests_fitnesses[i] = fitnesses[i]

                if fitnesses[i] < local_best_fitness:
                    local_best = position.copy()
                    local_best_fitness = fitnesses[i]
                    print(local_best_fitness, local_best)

        population = [tuple(positions[i]) for i in range(population_size)]
        fitness_by_individual = {tuple(positions[i]): fitnesses[i] for i in range(population_size)}

        observe_fn(population, fitness_by_individual)
        if terminate_fn(population, fitness_by_individual):
            break

    return min(population, key=fitness_by_individual.get)
