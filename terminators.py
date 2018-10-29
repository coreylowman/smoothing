import math


class GenerationTerminator:
    def __init__(self, num_generations):
        self.num_generations = num_generations
        self.generation = 0

    def __call__(self, population, fitness_by_individual) -> bool:
        self.generation += 1
        return self.generation >= self.num_generations


class NoImprovementsTerminator:
    def __init__(self, num_generations, epsilon):
        self.num_generations = num_generations
        self.generations_since_last_improvement = 0
        self.epsilon = epsilon

        self.bests = []

    def __call__(self, population, fitness_by_individual) -> bool:
        best_individual = min(population, key=fitness_by_individual.get)
        fitness = fitness_by_individual[best_individual]
        self.bests.append(fitness)

        recents = self.bests[-self.num_generations:]
        avg_recent = sum(recents) / len(recents)
        diffs = [abs(r - avg_recent) for r in recents]
        total_diffs = sum(diffs)

        return total_diffs < self.epsilon and len(self.bests) > self.num_generations
