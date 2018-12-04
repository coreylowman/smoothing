from typing import Callable, List, Tuple, Dict
import random
from math import log, sqrt, exp
import numpy
from functools import reduce
import operator
from terminators import GenerationTerminator, Any


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def clamp(v, min, max):
    if v < min:
        v = min
    elif v > max:
        v = max
    return v


Individual = Tuple[float]
Population = List[Individual]
PopulationFitness = Dict[Individual, float]

TerminateFn = Callable[[Population, PopulationFitness], bool]
FitnessFn = Callable[[Individual], float]
ObserveFn = Callable[[Population, PopulationFitness], None]
StepSizeFn = Callable[[], float]
AlgorithmFn = Callable[[TerminateFn, FitnessFn, List[float], List[float], ObserveFn, StepSizeFn], PopulationFitness]


def subspace_search(alg_fn: AlgorithmFn, gens=80) -> AlgorithmFn:
    def inner(
            terminate_fn: Callable[[Population, PopulationFitness], bool],
            fitness_fn: Callable[[Individual], float],
            mins: List[float], maxs: List[float],
            observe_fn: ObserveFn = None,
            step_size_fn: StepSizeFn = None,
    ) -> PopulationFitness:
        D = len(mins)

        while True:
            fitness_by_individual = alg_fn(Any(terminate_fn, GenerationTerminator(gens)), fitness_fn, mins, maxs,
                                           observe_fn, step_size_fn)
            population = list(fitness_by_individual.keys())
            if terminate_fn(population, fitness_by_individual):
                return fitness_by_individual

            mins = list(population[0][:D])
            maxs = list(population[0][:D])

            for individual in fitness_by_individual:
                for d in range(D):
                    if individual[d] < mins[d]:
                        mins[d] = individual[d]
                    if individual[d] > maxs[d]:
                        maxs[d] = individual[d]

    return inner


def genetic_algorithm(
        terminate_fn: TerminateFn, fitness_fn: FitnessFn, mins: List[float], maxs: List[float],
        observe_fn: ObserveFn = None, step_size_fn: StepSizeFn = None,
        population_size=100, crossover_rate=0.2, tournament_size=5,
) -> PopulationFitness:
    D = len(mins)
    mutation_rate = 1 / D

    def tournament_selection(population, fitness_by_individual) -> Individual:
        individuals = random.sample(population, tournament_size)
        return min(individuals, key=fitness_by_individual.get)

    def arithmetic_crossover(parent1, parent2) -> List[Individual]:
        if random.uniform(0, 1) < crossover_rate:
            a = random.uniform(0, 1)
            return [
                tuple(float(a * parent1[d] + (1 - a) * parent2[d]) for d in range(D)),
                tuple(float((1 - a) * parent1[d] + a * parent2[d]) for d in range(D)),
            ]
        else:
            return [parent1, parent2]

    def one_point_crossover(parent1, parent2) -> List[Individual]:
        if random.uniform(0, 1) < crossover_rate:
            crossover_pt = random.randrange(0, len(parent1))
            return [
                parent1[:crossover_pt] + parent2[crossover_pt:],
                parent2[:crossover_pt] + parent1[crossover_pt:],
            ]
        else:
            return [parent1, parent2]

    def creep_mutation(individual: Individual) -> Individual:
        def mutate(d, bit: float) -> float:
            if random.uniform(0, 1) <= mutation_rate:
                noise = random.gauss(0, 1.0)
                # noise += step_size_fn() * noise
                return bit + noise
            else:
                return bit

        return tuple(map(lambda d_b: mutate(*d_b), enumerate(individual)))

    population = []
    for i in range(population_size):
        population.append(tuple(random.uniform(mins[d], maxs[d]) for d in range(D)))

    fitness_by_individual = {individual: fitness_fn(individual) for individual in population}

    while True:
        all_children = set()
        while len(all_children) < population_size:
            # selection
            parent1 = tournament_selection(population, fitness_by_individual)
            parent2 = tournament_selection(population, fitness_by_individual)

            # reproduce
            children = arithmetic_crossover(parent1, parent2)

            # mutation
            for child in children:
                child = creep_mutation(child)
                all_children.add(child)
                if child not in fitness_by_individual:
                    fitness_by_individual[child] = fitness_fn(child)

        population = list(all_children)
        fitness_by_individual = {individual: fitness_by_individual[individual] for individual in population}

        if observe_fn:
            observe_fn(population, fitness_by_individual)
        if terminate_fn(population, fitness_by_individual):
            break

    return fitness_by_individual


def evolution_strategy(
        terminate_fn: TerminateFn, fitness_fn: FitnessFn, mins: List[float], maxs: List[float],
        observe_fn: ObserveFn = None, step_size_fn: StepSizeFn = None,
        mu=30, delta=200,
) -> PopulationFitness:
    """
    We used the (30; 200) ES with
    # self-adaptation of standard deviations in each dimension
        sigma = sigma * exp(tau_prime * gauss(0, 1) + tau * gauss_i(0, 1))
    # no correlated mutations,
    # discrete re-combination on object variables
    # global intermediate recombination on standard deviations
    # the standard deviations are initialized to a value of 3.0
    """
    D = len(mins)

    tau = 1 / sqrt(2 * sqrt(D))
    tau_prime = 1 / sqrt(2 * D)

    population = []
    deviation_by_individual = {}
    for i in range(mu):
        population.append(tuple(random.uniform(mins[d], maxs[d]) for d in range(D)))
        deviation_by_individual[population[-1]] = 3

    fitness_by_individual = {individual: fitness_fn(individual) for individual in population}

    while True:
        all_children = []
        while len(all_children) != delta:
            child = [0.0] * D

            n = random.gauss(0, 1)

            # global intermediate recombination on standard deviations
            dev_parent1 = deviation_by_individual[random.choice(population)]
            dev_parent2 = deviation_by_individual[random.choice(population)]
            deviation = dev_parent1 + random.uniform(0, 1) * (dev_parent2 - dev_parent1)

            # mutate standard deviation
            deviation *= exp(tau_prime * n + tau * random.gauss(0, 1))

            obj_parent1 = random.choice(population)
            obj_parent2 = random.choice(population)

            for d in range(D):
                # discrete re-combination on object variable
                child[d] = obj_parent1[d] if random.uniform(0, 1) <= 0.5 else obj_parent2[d]

                # mutate individual
                child[d] += deviation * random.gauss(0, 1)
                child[d] = clamp(child[d], mins[d], maxs[d])

            child = tuple(child)
            all_children.append(child)
            fitness_by_individual[child] = fitness_fn(child)
            deviation_by_individual[child] = deviation

        population = sorted(all_children, key=fitness_by_individual.get)[:mu]
        fitness_by_individual = {individual: fitness_by_individual[individual] for individual in population}
        deviation_by_individual = {individual: deviation_by_individual[individual] for individual in population}

        if observe_fn:
            observe_fn(population, fitness_by_individual)
        if terminate_fn(population, fitness_by_individual):
            break

    return fitness_by_individual


def differential_evolution(
        terminate_fn: TerminateFn, fitness_fn: FitnessFn, mins: List[float], maxs: List[float],
        observe_fn: ObserveFn = None, step_size_fn: StepSizeFn = None,
        population_size=20, crossover_rate=0.2, f=0.5,
) -> PopulationFitness:
    """Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces"""

    population = []
    fitnesses = []

    D = len(mins)

    for i in range(population_size):
        individual = numpy.zeros(D)
        for d in range(D):
            individual[d] = random.uniform(mins[d], maxs[d])
        population.append(individual)
        fitnesses.append(fitness_fn(tuple(individual)))

    while True:
        for i, individual in enumerate(population):
            other_individuals = set(range(population_size)) - {i}
            r1, r2, r3 = random.sample(other_individuals, 3)
            rj = random.randrange(0, D)
            x_r1, x_r2, x_r3 = population[r1], population[r2], population[r3]

            mutant = x_r1 + f * (x_r2 - x_r3)
            trial = individual.copy()
            for j in range(D):
                if random.uniform(0, 1) <= crossover_rate or j == rj:
                    trial[j] = mutant[j]

            for d in range(D):
                trial[d] = clamp(trial[d], mins[d], maxs[d])

            fitness = fitness_fn(trial)
            if fitness < fitnesses[i]:
                population[i] = trial
                fitnesses[i] = fitness

        pop = [tuple(population[i]) for i in range(population_size)]
        fitness_by_individual = {tuple(population[i]): fitnesses[i] for i in range(population_size)}

        if observe_fn:
            observe_fn(pop, fitness_by_individual)
        if terminate_fn(pop, fitness_by_individual):
            break

    return fitness_by_individual


def particle_swarm_optimization(
        terminate_fn: TerminateFn, fitness_fn: FitnessFn, mins: List[float], maxs: List[float],
        observe_fn: ObserveFn = None, step_size_fn: StepSizeFn = None,
        population_size=40, c1=0.5 + log(2), c2=0.5 + log(2), omega=1 / (2 * log(2)),
) -> PopulationFitness:
    """Standard Particle Swarm Optimisation 2011 at CEC-2013: A baseline for future PSO improvements"""

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
        fitnesses.append(fitness_fn(tuple(position)))
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
                velocity[d] = clamp(velocity[d], -maxs[d], maxs[d])

            position += velocity

            for d in range(D):
                if position[d] < mins[d]:
                    position[d] = mins[d]
                    velocity[d] = 0
                elif position[d] > maxs[d]:
                    position[d] = maxs[d]
                    velocity[d] = 0

            fitnesses[i] = fitness_fn(tuple(position))

            if fitnesses[i] < personal_bests_fitnesses[i]:
                personal_bests[i] = position.copy()
                personal_bests_fitnesses[i] = fitnesses[i]

                if fitnesses[i] < local_best_fitness:
                    local_best = position.copy()
                    local_best_fitness = fitnesses[i]

        population = [tuple(positions[i]) for i in range(population_size)]
        fitness_by_individual = {tuple(positions[i]): fitnesses[i] for i in range(population_size)}

        if observe_fn:
            observe_fn(population, fitness_by_individual)
        if terminate_fn(population, fitness_by_individual):
            break

    return fitness_by_individual
