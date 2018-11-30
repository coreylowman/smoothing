class Any:
    def __init__(self, *terminators):
        self.terminators = terminators

    def __call__(self, population, fitness_by_individual) -> bool:
        return any(map(lambda t: t(population, fitness_by_individual), self.terminators))


class GenerationTerminator:
    def __init__(self, num_generations):
        self.num_generations = num_generations
        self.generation = 0

    def __call__(self, population, fitness_by_individual) -> bool:
        self.generation += 1
        return self.generation >= self.num_generations


class ConvergenceTerminator:
    def __init__(self, num_generations, epsilon):
        self.num_generations = num_generations
        self.epsilon = epsilon

        self.bests = []

    def __call__(self, population, fitness_by_individual) -> bool:
        self.bests.append(min(fitness_by_individual.values()))
        if len(self.bests) <= self.num_generations:
            return False

        recents = self.bests[-self.num_generations:]
        diffs = [recents[i] - recents[i + 1] for i in range(self.num_generations - 1)]
        avg_diff = sum(diffs) / len(recents)

        return avg_diff < self.epsilon
