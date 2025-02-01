class EvolutionaryAlgorithm:
    def __init__(self, chromosomes, fitness_function, population_size, generations, mutation_rate, no_of_offsprings):
        self.chromosomes = chromosomes
        self.chromosomes_size = len(chromosomes)
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.no_of_offsprings = no_of_offsprings