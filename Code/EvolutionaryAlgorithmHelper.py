import random

class Chromosome:
    def __init__(self, index, genes, fitness):
        self.id = index
        self.genes = genes
        self.fitness = fitness
        

class EvolutionaryAlgorithmHelper:
    @staticmethod
    def fitness_proportionate_selection(generation, k, optimize_max=False):
        """
        Selects k parents/survivor using Fitness Proportionate Selection (Roulette Wheel Selection).
        
        Args:
            generation (list): A list of Chromosome objects representing the current generation.
            k (int): Number of parents/survivor to select.
            optimize_max (bool): Whether to optimize for maximum fitness (default: False).
        
        Returns:
            list: A list of k selected Chromosome objects as parents/survivor.
        """
        if k > len(generation):
            raise ValueError("k cannot be greater than the generation size.")
        
        fitness_values = [chromosome.fitness for chromosome in generation]
        if not optimize_max:
            fitness_values = [1 / (f + 1e-10) for f in fitness_values]  # Invert for minimization
        total_fitness = sum(fitness_values)
        if total_fitness <= 0:
            return EvolutionaryAlgorithmHelper.random_selection(generation, k)
        probabilities = [fitness / total_fitness for fitness in fitness_values]
        parents = [generation[EvolutionaryAlgorithmHelper.roulette_wheel_selection(probabilities)] for _ in range(k)]
        return parents

    @staticmethod
    def roulette_wheel_selection(probabilities):
        """
        Performs Roulette Wheel Selection based on given probabilities.
        
        Args:
            probabilities (list): A list of probabilities corresponding to each chromosome.
        
        Returns:
            int: Index of the selected chromosome.
        """
        r = random.random()
        cumulative = 0.0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return i
        return len(probabilities) - 1
    
    @staticmethod
    def rank_based_selection(generation, k, optimize_max=False):
        """
        Selects k parents/survivor using Rank-Based Selection.
        
        Args:
            generation (list): A list of Chromosome objects representing the current generation.
            k (int): Number of parents/survivor to select.
            optimize_max (bool): Whether to optimize for maximum fitness (default: False).
        
        Returns:
            list: A list of k selected Chromosome objects as parents/survivor.
        """
        if k > len(generation):
            raise ValueError("k cannot be greater than the generation size.")
        
        generation_copy = generation.copy()
        generation_copy.sort(key=lambda x: x.fitness, reverse=optimize_max)
        n = len(generation)
        fitness_values = [n - i for i in range(n)]
        total_fitness = sum(fitness_values)
        probabilities = [fitness / total_fitness for fitness in fitness_values]
        parents = [generation_copy[EvolutionaryAlgorithmHelper.roulette_wheel_selection(probabilities)] for _ in range(k)]
        return parents
    
    @staticmethod
    def binary_tournament_selection(generation, k, optimize_max=False):
        """
        Selects k parents/survivor using Binary Tournament Selection.
        
        Args:
            generation (list): A list of Chromosome objects representing the current generation.
            k (int): Number of parents/survivor to select.
            optimize_max (bool): Whether to optimize for maximum fitness (default: False).
        
        Returns:
            list: A list of k selected Chromosome objects as parents/survivor.
        """
        if k > len(generation):
            raise ValueError("k cannot be greater than the generation size.")
        
        def select_parent():
            candidates = random.sample(generation, 2)
            return max(candidates, key=lambda x: x.fitness) if optimize_max else min(candidates, key=lambda x: x.fitness)
        
        parents = [select_parent() for _ in range(k)]
        return parents
    
    @staticmethod
    def truncation_selection(generation, k, optimize_max=False):
        """
        Selects k parents/survivor using Truncation Selection.
        
        Args:
            generation (list): A list of Chromosome objects representing the current generation.
            k (int): Number of parents/survivor to select.
            optimize_max (bool): Whether to optimize for maximum fitness (default: False).
        
        Returns:
            list: A list of k selected Chromosome objects as parents/survivor.
        """
        if k > len(generation):
            raise ValueError("k cannot be greater than the generation size.")
        
        generation_copy = generation.copy()
        generation_copy.sort(key=lambda x: x.fitness, reverse=optimize_max)
        return generation_copy[:k]
    
    @staticmethod
    def random_selection(generation, k):
        """
        Selects k parents/survivor randomly.
        
        Args:
            generation (list): A list of Chromosome objects representing the current generation.
            k (int): Number of parents/survivor to select.
        
        Returns:
            list: A list of k randomly selected Chromosome objects as parents/survivor.
        """
        if k > len(generation):
            raise ValueError("k cannot be greater than the generation size.")
        
        return random.sample(generation, k)
