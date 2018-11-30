from algorithms import Individual
from typing import Callable
import random
import numpy


def sphere_sample(x, sample_area_size):
    n = len(x)
    length = sample_area_size * (random.uniform(0, 1) ** (1 / n))
    offset = numpy.random.uniform(-1, 1, n)
    offset = length * offset / numpy.linalg.norm(offset)
    return tuple(x + offset)


def gaussian_sample(x, sample_area_size):
    offset = numpy.random.normal(0, sample_area_size, size=len(x))
    return tuple(x + offset)


class Smoother:
    def __init__(self,
                 fitness_fn: Callable[[Individual], float],
                 sample_fn: Callable[[Individual, float], Individual],
                 size: float, num_points: int,
                 reduction_pct: float, reduction_frequency: int,
                 on=True):
        self._fitness_fn = fitness_fn

        self._sample_fn = sample_fn
        self._sample_area_size = size
        self._num_sample_points = num_points
        self._sample_area_reduction_pct = reduction_pct
        self._sample_reduction_frequency = reduction_frequency
        self._generations = 0

        self.function_evaluations = 0
        self.total_generations = 0

        self._fn = self._smoothed_fitness_fn if on else self._normal_fitness_fn

    def __call__(self, x):
        return self._fn(x)

    def _normal_fitness_fn(self, x):
        self.function_evaluations += 1
        return self._fitness_fn(x)

    def _smoothed_fitness_fn(self, x):
        if self._sample_area_size <= 1e-8:
            return self._normal_fitness_fn(x)
        points = [self._sample_fn(x, self._sample_area_size) for _ in range(self._num_sample_points)]
        points.append(x)
        return sum(map(self._normal_fitness_fn, points)) / len(points)

    def observe(self, population, fitness_by_individual):
        self._generations += 1
        self.total_generations += 1
        if self._generations >= self._sample_reduction_frequency:
            self._sample_area_size *= (1 - self._sample_area_reduction_pct)
            self._generations = 0

        fitnesses = list(map(self._normal_fitness_fn, population))
        print(min(fitness_by_individual.values()), min(fitnesses), sum(fitnesses) / len(fitness_by_individual),
              self._generations, self._sample_area_size, self.total_generations)
