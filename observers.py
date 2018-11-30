import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NothingObserver:
    def __init__(self):
        pass

    def __call__(self, population, fitness_by_individual):
        pass


class ChainedObserver:
    def __init__(self, *observers):
        self.observers = observers

    def __call__(self, population, fitness_by_individual):
        for obs in self.observers:
            obs(population, fitness_by_individual)


class GenerationPrinter:
    def __init__(self):
        self.generation = 0

    def __call__(self, population, fitness_by_individual):
        self.generation += 1
        print(self.generation)


class BestIndividualPrinter:
    def __init__(self):
        pass

    def __call__(self, population, fitness_by_individual):
        best_individual = min(population, key=fitness_by_individual.get)
        print(fitness_by_individual[best_individual], best_individual)


class Population3DPlotter:
    def __init__(self, f, mins, maxs, multiplier=10):
        self.f = f
        x = np.linspace(mins[0], maxs[0], multiplier * (maxs[0] - mins[0]))
        y = np.linspace(mins[0], maxs[0], multiplier * (maxs[1] - mins[1]))
        self.X, self.Y = np.meshgrid(x, y)

        self.Z = self.X.copy()
        for i in range(len(self.X)):
            for j in range(len(self.X[i])):
                self.Z[i, j] = f((self.X[i, j], self.Y[i, j]))

        plt.ion()
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        self.ax = fig.gca(projection='3d')
        self.ax.view_init(90, 0)

    def __call__(self, population, fitness_by_individual):
        self.ax.clear()
        self.ax.plot_surface(self.X, self.Y, self.Z, cmap='coolwarm')

        xs = list(map(lambda i: i[0], population))
        ys = list(map(lambda i: i[1], population))
        self.ax.plot(xs, ys, 'kx', markersize=1)

        plt.pause(0.0001)
