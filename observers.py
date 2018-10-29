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
