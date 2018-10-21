from typing import Callable, List, Tuple, Dict
import random

Individual = Tuple[float]
Population = List[Individual]
PopulationFitness = Dict[Individual, float]


def genetic_algorithm(
        terminate_fn: Callable[[Population], bool],
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

        population = replace_fn(population, all_children, fitness_by_individual)
        fitness_by_individual = {individual: fitness_by_individual[individual] for individual in population}

        if terminate_fn(population):
            break

    return population


def differential_evolution(
        terminate_fn: Callable[[Population], bool],
        initialize_fn: Callable[[], Population],
        fitness_fn: Callable[[Individual], float],
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

            if fitness_by_individual[new_individual] < fitness_by_individual[individual]:
                new_population.append(new_individual)
            else:
                new_population.append(individual)

        fitness_by_individual = {individual: fitness_by_individual[individual] for individual in new_population}
        population = new_population

        if terminate_fn(population):
            break

    return population


def particle_swarm_optimization(
        terminate_fn: Callable[[Population], bool],
        initialize_fn: Callable[[], Population],
        velocity_initialize_fn: [Callable[], Tuple[float]],
        fitness_fn: Callable[[Individual], float],
        current_weight: float = 1.0, previous_weight: float = 2.0, global_weight: float = 2.0,
) -> Population:
    population = initialize_fn()
    fitness_by_individual = {individual: fitness_fn(individual) for individual in population}

    best_by_particle = {i: individual for i, individual in enumerate(population)}
    velocity_by_particle = {i: velocity_initialize_fn() for i, individual in enumerate(population)}

    global_best = max(population, key=fitness_by_individual.get)

    while True:
        for particle in velocity_by_particle:
            position = population[particle]
            previous_best = best_by_particle[particle]
            velocity = velocity_by_particle[particle]

            new_velocity = []
            for d in range(len(particle)):
                r_p = random.uniform(0, 1)
                r_g = random.uniform(0, 1)

                current_term = current_weight * velocity[d]
                previous_term = previous_weight * r_p * (previous_best[d] - position[d])
                global_term = global_weight * r_g * (global_best[d] - position[d])

                new_velocity.append(current_term + previous_term + global_term)
            velocity = tuple(new_velocity)
            velocity_by_particle[particle] = velocity

            new_position = []
            for d in range(len(particle)):
                new_position.append(position[d] + velocity[d])
            new_position = tuple(new_position)

            fitness_by_individual[new_position] = fitness_fn(new_position)
            if fitness_by_individual[new_position] < fitness_by_individual[previous_best]:
                best_by_particle[particle] = new_position
                if fitness_by_individual[new_position] < fitness_by_individual[global_best]:
                    global_best = new_position
            population[particle] = new_position

        if terminate_fn(population):
            break

    return population
