from unittest import TestCase
from typing import List
from algorithms import genetic_algorithm, Individual
from terminators import *
from observers import *
from functions import *


class GATestCase(TestCase):
    def test_simple_real(self):
        def initialize(population_size=25, individual_size=5) -> List[Individual]:
            return [tuple(float(random.uniform(0, 1)) for _ in range(individual_size)) for _ in range(population_size)]

        def select(population, fitness_by_individual, number=2) -> List[Individual]:
            max_fitness = max(fitness_by_individual.values())
            weights = [max_fitness - fitness_by_individual[individual] for individual in population]
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
            # return select(old_population + new_population, fitness_by_individual, number=len(old_population))
            return new_population

        genetic_algorithm(NoImprovementsTerminator(10, 0.01), BestIndividualPrinter(), initialize, YaoLiuLin.f1, select,
                          crossover, mutation, replace)


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

    def test_f12(self):
        self.assertAlmostEqual(YaoLiuLin.f12((-1,) * 30), 0)

    def test_f13(self):
        self.assertAlmostEqual(YaoLiuLin.f13((1,) * 30), 0)

    def test_f14(self):
        self.assertAlmostEqual(YaoLiuLin.f14((-32, -32)), 1, places=2)

    def test_f15(self):
        self.assertAlmostEqual(YaoLiuLin.f15((0.1928, 0.1908, 0.1231, 0.1358)), 0.0003075)

    def test_f16(self):
        self.assertAlmostEqual(YaoLiuLin.f16((0.08983, -0.7126)), -1.0316285, places=6)
        self.assertAlmostEqual(YaoLiuLin.f16((-0.08983, 0.7126)), -1.0316285, places=6)

    def test_f17(self):
        self.assertAlmostEqual(YaoLiuLin.f17((-3.142, 12.275)), 0.398, places=3)
        self.assertAlmostEqual(YaoLiuLin.f17((3.142, 2.275)), 0.398, places=3)
        # self.assertAlmostEqual(YaoLiuLin.f17((9.425, 2.425)), 0.398, places=3)

    def test_f18(self):
        self.assertEqual(YaoLiuLin.f18((0, -1)), 3)

    def test_f19(self):
        self.assertAlmostEqual(YaoLiuLin.f19((0.114, 0.556, 0.852)), -3.86, places=2)

    def test_f20(self):
        self.assertAlmostEqual(YaoLiuLin.f20((0.201, 0.150, 0.477, 0.275, 0.311, 0.657)), -3.32, places=2)

    def test_f21_f22(self):
        c = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
        a = [
            [4, 4, 4, 4],
            [1, 1, 1, 1],
            [8, 8, 8, 8],
            [6, 6, 6, 6],
            [3, 7, 3, 7],
            [2, 9, 2, 9],
            [5, 5, 3, 3],
            [8, 1, 8, 1],
            [6, 2, 6, 2],
            [7, 3.6, 7, 3.6],
        ]
        for i in range(5):
            self.assertAlmostEqual(YaoLiuLin.f21(a[i]), -1 / c[i], places=0)

        for i in range(7):
            self.assertAlmostEqual(YaoLiuLin.f22(a[i]), -1 / c[i], places=0)
