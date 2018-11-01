from typing import List
from algorithms import genetic_algorithm, Individual
from terminators import *
from observers import *
from functions import *


def fitness_proportionate_select(population, fitness_by_individual, number=2) -> List[Individual]:
    max_fitness = max(fitness_by_individual.values())
    weights = [max_fitness - fitness_by_individual[individual] for individual in population]
    return random.choices(population, weights, k=number)


def one_point_crossover(parents, crossover_rate=0.1) -> List[Individual]:
    if random.uniform(0, 1) < crossover_rate:
        parent1, parent2 = parents
        crossover_pt = random.randrange(0, len(parent1))
        return [
            parent1[:crossover_pt] + parent2[crossover_pt:],
            parent2[:crossover_pt] + parent1[crossover_pt:],
        ]
    else:
        return parents


def bit_mutation(individual: Individual, mutation_rate=0.1, epsilon=0.1) -> Individual:
    def mutate(bit: float) -> float:
        return bit + random.uniform(-epsilon, epsilon)

    return tuple(mutate(bit) if random.uniform(0, 1) <= mutation_rate else bit for bit in individual)


def new_generation_replace(old_population, new_population, fitness_by_individual) -> List[Individual]:
    # return select(old_population + new_population, fitness_by_individual, number=len(old_population))
    return new_population


terminator = NoImprovementsTerminator(5, 0.01)

for f in [YaoLiuLin.f1]:
    n = get_dimensions(f)
    mins, maxs = get_bounds(f)


    def initialize(population_size=25) -> List[Individual]:
        return [tuple(float(random.uniform(mins[i], maxs[i])) for i in range(n)) for _ in range(population_size)]


    genetic_algorithm(terminator, BestIndividualPrinter(), initialize, f, fitness_proportionate_select,
                      one_point_crossover, bit_mutation, new_generation_replace)
