import random
from unittest import TestCase
from typing import List, Tuple
from algorithms import genetic_algorithm
from functions import *

BinaryIndividual = Tuple[int]
RealIndividual = Tuple[float]


class GATestCase(TestCase):
    def test_simple_binary(self):
        def initialize(population_size=4, individual_size=5) -> List[BinaryIndividual]:
            return [tuple(random.choice([0, 1]) for _ in range(individual_size)) for _ in range(population_size)]

        def fitness(individual) -> float:
            return sum(individual) ** 2

        def select(population, fitness_by_individual, number=2) -> List[BinaryIndividual]:
            weights = [fitness_by_individual[individual] for individual in population]
            return random.choices(population, weights, k=number)

        def crossover(parents, crossover_rate=0.1) -> List[BinaryIndividual]:
            if random.uniform(0, 1) < crossover_rate:
                parent1, parent2 = parents
                crossover_pt = random.randrange(0, len(parent1))
                return [
                    parent1[:crossover_pt] + parent2[crossover_pt:],
                    parent2[:crossover_pt] + parent1[crossover_pt:],
                ]
            else:
                return parents

        def mutation(individual: BinaryIndividual, mutation_rate=0.1) -> BinaryIndividual:
            return tuple(int(not bit) if random.uniform(0, 1) <= mutation_rate else bit for bit in individual)

        def replace(old_population, new_population, fitness_by_individual) -> List[BinaryIndividual]:
            return select(old_population + new_population, fitness_by_individual, number=len(old_population))

        genetic_algorithm(50, initialize, fitness, select, crossover, mutation, replace)

    def test_simple_real(self):
        def initialize(population_size=4, individual_size=5) -> List[RealIndividual]:
            return [tuple(random.uniform(0, 1) for _ in range(individual_size)) for _ in range(population_size)]

        def fitness(individual) -> float:
            return sum(individual) ** 2

        def select(population, fitness_by_individual, number=2) -> List[RealIndividual]:
            weights = [fitness_by_individual[individual] for individual in population]
            return random.choices(population, weights, k=number)

        def crossover(parents, crossover_rate=0.1) -> List[RealIndividual]:
            if random.uniform(0, 1) < crossover_rate:
                parent1, parent2 = parents
                crossover_pt = random.randrange(0, len(parent1))
                return [
                    parent1[:crossover_pt] + parent2[crossover_pt:],
                    parent2[:crossover_pt] + parent1[crossover_pt:],
                ]
            else:
                return parents

        def mutation(individual: RealIndividual, mutation_rate=0.1, epsilon=0.1) -> RealIndividual:
            def mutate(bit: float) -> float:
                return bit + random.uniform(-epsilon, epsilon)

            return tuple(mutate(bit) if random.uniform(0, 1) <= mutation_rate else bit for bit in individual)

        def replace(old_population, new_population, fitness_by_individual) -> List[RealIndividual]:
            return select(old_population + new_population, fitness_by_individual, number=len(old_population))

        genetic_algorithm(50, initialize, fitness, select, crossover, mutation, replace)


class YaoLiuLinTestCase(TestCase):
    def test_f1(self):
        self.assertEqual(YaoLiuLin.f1((0,) * 30), 0)

    def test_f2(self):
        self.assertEqual(YaoLiuLin.f2((0,) * 30), 0)

    def test_f3(self):
        self.assertEqual(YaoLiuLin.f3((0,) * 30), 0)

    def test_f4(self):
        self.assertEqual(YaoLiuLin.f4((0,) * 30), 0)

    def test_f5(self):
        self.assertEqual(YaoLiuLin.f5((1,) * 30), 0)

    def test_f6(self):
        self.assertEqual(YaoLiuLin.f6((0,) * 30), 0)

    def test_f7(self):
        self.assertEqual(YaoLiuLin.f7((0,) * 30), 0)

    def test_f8(self):
        self.assertAlmostEqual(YaoLiuLin.f8((420.9687,) * 30), -12569.5, places=1)

    def test_f9(self):
        self.assertEqual(YaoLiuLin.f9((0,) * 30), 0)

    def test_f10(self):
        self.assertAlmostEqual(YaoLiuLin.f10((0,) * 30), 0)

    def test_f11(self):
        self.assertEqual(YaoLiuLin.f11((0,) * 30), 0)

    def test_f18(self):
        self.assertEqual(YaoLiuLin.f18((0, -1)), 3)
