import random
from unittest import TestCase
from typing import List, Tuple
from .algorithms import genetic_algorithm, Individual


class GenerationTerminator:
    def __init__(self, num_generations):
        self.num_generations = num_generations
        self.generation = 0

    def test(self, old_pop, new_pop) -> bool:
        self.generation += 1
        return self.generation >= self.num_generations


class GATestCase(TestCase):
    def test_simple_real(self):
        term = GenerationTerminator(50)

        def initialize(population_size=4, individual_size=5) -> List[Individual]:
            return [tuple(float(random.uniform(0, 1)) for _ in range(individual_size)) for _ in range(population_size)]

        def fitness(individual) -> float:
            return sum(individual) ** 2

        def select(population, fitness_by_individual, number=2) -> List[Individual]:
            weights = [fitness_by_individual[individual] for individual in population]
            return random.choices(population, weights, k=number)

        def crossover(parents, crossover_rate=0.1) -> List[Individual]:
            if random.uniform(0, 1) < crossover_rate:
                parent1, parent2 = parents
                crossover_pt = random.randrange(0, len(parent1))
                return [
                    parent1[:crossover_pt] + parent2[crossover_pt:],
                    parent2[:crossover_pt] + parent1[crossover_pt:],
                ]
            else:
                return parents

        def mutation(individual: Individual, mutation_rate=0.1, epsilon=0.1) -> Individual:
            def mutate(bit: float) -> float:
                return bit + random.uniform(-epsilon, epsilon)

            return tuple(mutate(bit) if random.uniform(0, 1) <= mutation_rate else bit for bit in individual)

        def replace(old_population, new_population, fitness_by_individual) -> List[Individual]:
            return select(old_population + new_population, fitness_by_individual, number=len(old_population))

        genetic_algorithm(term.test, initialize, fitness, select, crossover, mutation, replace)
