from algorithms import Individual
from typing import Callable
import random
from math import sqrt


def random_unit_vector(n):
    vec = [random.gauss(0, 1) for _ in range(n)]
    mag = sqrt(sum(x ** 2 for x in vec))
    return tuple(x / mag for x in vec)


def random_neighbor_of(x, r):
    n = len(x)
    offset = random_unit_vector(n)
    return tuple(x[i] + r * offset[i] for i in range(n))


class SphereSmoother:
    def __init__(self, fitness_fn: Callable[[Individual], float],
                 radius: float, reduce_pct: float,
                 num_points: int):
        self.fitness_fn = fitness_fn

        self.num_points = num_points
        self.radius = radius
        self.reduce_pct = reduce_pct

    def smoothed_fitness_fn(self, x):
        points = [random_neighbor_of(x, self.radius) for _ in range(self.num_points)]
        points.append(x)
        return sum(map(self.fitness_fn, points)) / len(points)

    def reduce_size(self):
        self.radius = self.reduce_pct * self.radius
