import sys
import random
import math
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('..'))
from EvolutionaryAlgorithmHelper import Chromosome, EvolutionaryAlgorithmHelper


class TravelSalesmanProblemSolver:
    def __init__(self, dataset, population_size, no_of_offsprings, no_of_generations, mutation_rate, no_of_iterations):
        self.dataset = dataset
        self.population_size = population_size
        self.no_of_offsprings = no_of_offsprings
        self.no_of_generations = no_of_generations
        self.mutation_rate = mutation_rate
        self.no_of_iterations = no_of_iterations

    @staticmethod
    def calculate_fitness(genes, dataset):
        """
        Calculate the fitness of a chromosome using Euclidean distance.
        
        Args:
            genes (list): List of indices representing the order of cities.
            dataset (list): List of city coordinates.
        
        Returns:
            float: Total Euclidean distance for the given path.
        """
        total_distance = 0
        for i in range(len(genes) - 1):
            city1 = dataset[genes[i]]
            city2 = dataset[genes[i + 1]]
            total_distance += math.sqrt(sum((x - y) ** 2 for x, y in zip(city1, city2)))
        # Return to the starting city
        city1 = dataset[genes[-1]]
        city2 = dataset[genes[0]]
        total_distance += math.sqrt(sum((x - y) ** 2 for x, y in zip(city1, city2)))
        return total_distance

    def initialize_population(self):
        """
        Initialize the population by shuffling the dataset and creating chromosomes.
        
        Returns:
            list: A list of Chromosome objects.
        """
        population = []
        for i in range(1000):
            genes = list(range(len(self.dataset)))
            random.shuffle(genes)
            fitness = self.calculate_fitness(genes, self.dataset)
            population.append(Chromosome(i, genes, fitness))
        # Select top population_size chromosomes
        population.sort(key=lambda x: x.fitness)
        return population[:self.population_size]

    def crossover(self, parent1, parent2):
        """
        Perform Partially Mapped Crossover (PMX) to preserve gene relationships.

        Args:
            parent1 (Chromosome): The first parent.
            parent2 (Chromosome): The second parent.

        Returns:
            Chromosome: The offspring chromosome.
        """
        size = len(parent1.genes)
        start, end = sorted(random.sample(range(size), 2))
        
        # Initialize offspring with -1 (invalid genes)
        offspring_genes = [-1] * size
        
        # Copy a segment from parent1
        offspring_genes[start:end] = parent1.genes[start:end]
        
        # Fill remaining positions with genes from parent2
        for i in range(size):
            if offspring_genes[i] == -1:
                gene = parent2.genes[i]
                while gene in offspring_genes:
                    gene = parent2.genes[parent1.genes.index(gene)]
                offspring_genes[i] = gene

        fitness = self.calculate_fitness(offspring_genes, self.dataset)
        return Chromosome(-1, offspring_genes, fitness)

    def mutate(self, chromosome):
        """
        Perform 2-opt mutation to improve the tour by reversing a segment.

        Args:
            chromosome (Chromosome): The chromosome to mutate.

        Returns:
            Chromosome: The mutated chromosome.
        """
        if random.random() < self.mutation_rate:
            size = len(chromosome.genes)
            idx1, idx2 = sorted(random.sample(range(size), 2))
            
            # Reverse the segment between idx1 and idx2
            chromosome.genes[idx1:idx2] = reversed(chromosome.genes[idx1:idx2])
            
            chromosome.fitness = self.calculate_fitness(chromosome.genes, self.dataset)
        
        return chromosome


    def evolve(self):
        """
        Evolve the population over the specified number of generations and iterations.
        """
        avg_bsf = [0 for _ in range(self.no_of_iterations)]
        avg_fitness = [0 for _ in range(self.no_of_iterations)]
        for iteration in range(self.no_of_iterations):
            population = self.initialize_population()
            bsf = [0 for _ in range(self.no_of_generations)]
            avg_gen_fitness = [0 for _ in range(self.no_of_generations)]
            for generation in range(self.no_of_generations):
                offsprings = []
                for _ in range(self.no_of_offsprings):
                    parent1, parent2 = EvolutionaryAlgorithmHelper.random_selection(population, 2)
                    offspring = self.crossover(parent1, parent2)
                    offspring = self.mutate(offspring)
                    offsprings.append(offspring)
                population.extend(offsprings)
                population = EvolutionaryAlgorithmHelper.random_selection(population, self.population_size)
                # Get best fitness from current generation and add to the list
                bsf[generation] = population[0].fitness
                avg_gen_fitness[generation] = sum(chromosome.fitness for chromosome in population) / self.population_size
                if generation == self.no_of_generations - 1: 
                    # Check if the answer is valid
                    s = set(i for i in population[0].genes)
                    if len(s) != len(population[0].genes) and len(s) != len(self.dataset):
                        raise ValueError("Invalid answer")
                    avg_bsf[iteration] = bsf
                    avg_fitness[iteration] = avg_gen_fitness
        
        for i in range(self.no_of_generations):
            gen_avg_bsf = sum(avg_bsf[j][i] for j in range(self.no_of_iterations)) / self.no_of_iterations
            print(f"Generation {i + 1} Average Best: {gen_avg_bsf}")
            gen_avg_fitness = sum(avg_fitness[j][i] for j in range(self.no_of_iterations)) / self.no_of_iterations
            print(f"Generation {i + 1} Average Fitness: {gen_avg_fitness}")

        # *********** Plot the graph for Average Best Fitness ***********
        # Generate X-axis values from 1 to no_of_generations
        x = np.arange(1, self.no_of_generations + 1)

        # Compute Y-axis values (gen_avg_bsf) using NumPy
        gen_avg_bsf = np.mean(avg_bsf, axis=0)  # NumPy computes mean along iterations
        

        # Plot the graph for Average Best Fitness
        plt.figure(figsize=(12, 6))
        plt.plot(x, gen_avg_bsf, marker='o', markersize=2, linewidth=1, label="Avg Best Fitness")

        # Labeling
        plt.xlabel("Generation")
        plt.ylabel("Average Best Fitness")
        plt.title("TSP using EA - (Random and Random)")
        plt.legend()
        plt.grid(True)

        # Optimize visibility for large generation counts
        plt.xticks(np.linspace(1, self.no_of_generations, num=10, dtype=int))  # Show only a few ticks on X-axis
        plt.yticks(np.linspace(min(gen_avg_bsf), max(gen_avg_bsf), num=10))  # Adjust Y-ticks for clarity

        plt.show()

        # *********** Plot the graph for Average Fitness ***********
        # Compute Y-axis values (avg_fitness) using NumPy
        avg_fitness = np.mean(avg_fitness, axis=0)  # NumPy computes mean along iterations

        # Plot the graph for Average Fitness
        plt.figure(figsize=(12, 6))
        plt.plot(x, avg_fitness, marker='o', markersize=2, linewidth=1, label="Avg Fitness")

        # Labeling
        plt.xlabel("Generation")
        plt.ylabel("Average Fitness")
        plt.title("TSP using EA - (Random and Random)")
        plt.legend()
        plt.grid(True)

        # Optimize visibility for large generation counts
        plt.xticks(np.linspace(1, self.no_of_generations, num=10, dtype=int))
        plt.yticks(np.linspace(min(avg_fitness), max(avg_fitness), num=10))

        plt.show()

if __name__ == '__main__':
    sys.stdin = open('dataset.txt', 'r', encoding='utf-8')
    # sys.stdout = open('output.txt', 'w', encoding='utf-8')
    dataset = [list(map(float, line.split()))[1:] for line in sys.stdin]
    
    if len(sys.argv) != 6:
        print("Usage: python TSP.py <population_size> <no_of_offsprings> <no_of_generations> <mutation_rate> <no_of_iterations>")
        sys.exit(1)
    
    population_size = int(sys.argv[1])
    no_of_offsprings = int(sys.argv[2])
    no_of_generations = int(sys.argv[3])
    mutation_rate = float(sys.argv[4])
    no_of_iterations = int(sys.argv[5])

    solver = TravelSalesmanProblemSolver(
        dataset, 
        population_size, 
        no_of_offsprings, 
        no_of_generations, 
        mutation_rate, 
        no_of_iterations
    )
    solver.evolve()