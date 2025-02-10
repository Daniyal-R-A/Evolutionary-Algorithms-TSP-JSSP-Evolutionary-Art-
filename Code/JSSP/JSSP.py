import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from EvolutionaryAlgorithmHelper import Chromosome, EvolutionaryAlgorithmHelper

# Loading JSSP data
def load_jssp_data(filename):
    with open(f"Code/JSSP/{filename}.txt", "r") as file:
        lines = file.readlines()
    
    num_jobs, num_machines = map(int, lines[1].split())
    jobs = [[] for _ in range(num_jobs)]

    for i, line in enumerate(lines[2:num_jobs + 2]):  
        data = list(map(int, line.split()))
        jobs[i] = [(data[j], data[j + 1]) for j in range(0, len(data), 2)]
    
    return num_jobs, num_machines, jobs

# Decoding chromosome into schedule
def decode_schedule(chromosome, jobs, machines, job_operations):
    machine_time = [0] * machines
    job_time = [0] * jobs
    schedule = []
    
    for gene in chromosome.genes:
        job, op_idx = gene
        machine, proc_time = job_operations[job][op_idx]
        start_time = max(job_time[job], machine_time[machine])
        end_time = start_time + proc_time
        job_time[job] = end_time
        machine_time[machine] = end_time
        schedule.append((job, op_idx, machine, start_time, end_time))
    
    return schedule, max(job_time)

# Fitness function (minimizing makespan)
def fitness_function(chromosome, jobs, machines, job_operations):
    _, makespan = decode_schedule(chromosome, jobs, machines, job_operations)
    return makespan  

# Initializing population
def initialize_population(pop_size, jobs, job_operations):
    population = []
    for i in range(pop_size):
        genes = [(j, op) for j in range(jobs) for op in range(len(job_operations[j]))]
        random.shuffle(genes)
        chromosome = Chromosome(i, genes, 0)
        chromosome.fitness = fitness_function(chromosome, jobs, len(job_operations[0]), job_operations)
        population.append(chromosome)
    return population

def evolve_population(population, jobs, job_operations, mutation_rate=0.1, selection_method="FPS", survival_method="Random"):
    new_population = []
    pop_size = len(population)
    
    # Parent selection
    if selection_method == "FPS":
        parents = EvolutionaryAlgorithmHelper.fitness_proportionate_selection(population, pop_size // 2)
    elif selection_method == "BinaryTournament":
        parents = EvolutionaryAlgorithmHelper.binary_tournament_selection(population, pop_size // 2)
    elif selection_method == "Truncation":
        parents = EvolutionaryAlgorithmHelper.truncation_selection(population, pop_size // 2)
    else:  
        parents = EvolutionaryAlgorithmHelper.random_selection(population, pop_size // 2)

    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            parent1, parent2 = parents[i], parents[i+1]
            child1, child2 = crossover(parent1, parent2)
            mutation(child1, mutation_rate)
            mutation(child2, mutation_rate)
            child1.fitness = fitness_function(child1, jobs, len(job_operations[0]), job_operations)
            child2.fitness = fitness_function(child2, jobs, len(job_operations[0]), job_operations)
            new_population.extend([child1, child2])

    # Ensuring new_population is at least pop_size before applying survivor selection
    if len(new_population) < pop_size:
        deficit = pop_size - len(new_population)
        additional_individuals = random.choices(population, k=deficit)  # Fill the gap with random previous individuals
        new_population.extend(additional_individuals)

    # Survivor selection
    if survival_method == "Truncation":
        new_population.sort(key=lambda x: x.fitness)
        new_population = new_population[:pop_size]
    elif survival_method == "RBS":
        new_population = EvolutionaryAlgorithmHelper.rank_based_selection(new_population, pop_size)
    elif survival_method == "Random":
        random.shuffle(new_population)
        new_population = new_population[:pop_size]
    
    return new_population


# Doing Crossover (PMX) to mantain order
def crossover(parent1, parent2):
    size = len(parent1.genes)
    point1, point2 = sorted(random.sample(range(size), 2))
    child1_genes = parent1.genes[:point1] + parent2.genes[point1:point2] + parent1.genes[point2:]
    child2_genes = parent2.genes[:point1] + parent1.genes[point1:point2] + parent2.genes[point2:]
    return Chromosome(0, child1_genes, 0), Chromosome(1, child2_genes, 0)

# performing Mutation (Swap Mutation)
def mutation(chromosome, rate):
    if random.random() < rate:
        i, j = random.sample(range(len(chromosome.genes)), 2)
        chromosome.genes[i], chromosome.genes[j] = chromosome.genes[j], chromosome.genes[i]

# Now, we are running Evolutionary Algorithm with multiple iterations
def evolutionary_algorithm(filename, pop_size=100, generations=200, iterations=10, selection_method="FPS", survival_method="Random"):
    jobs, machines, job_operations = load_jssp_data(filename)

    avg_best_fitness = np.zeros(generations)
    avg_avg_fitness = np.zeros(generations)

    for it in range(iterations):
        print(f"\nIteration {it + 1} / {iterations}")
        population = initialize_population(pop_size, jobs, job_operations)
        best_fitness = []
        avg_fitness = []

        for generation in range(generations):
            population = evolve_population(population, jobs, job_operations, selection_method=selection_method, survival_method=survival_method)
            if not population:
                print(f"Warning: Population empty at generation {generation}, retaining previous.")
                population = initialize_population(pop_size, jobs, job_operations)

            population.sort(key=lambda x: x.fitness)
            best_fitness.append(population[0].fitness)
            avg_fitness.append(np.mean([c.fitness for c in population]))

        avg_best_fitness += np.array(best_fitness)
        avg_avg_fitness += np.array(avg_fitness)

    avg_best_fitness /= iterations
    avg_avg_fitness /= iterations

    plot_results(avg_best_fitness, avg_avg_fitness, selection_method, survival_method)

# At the end we are ploting the results
def plot_results(best_fitness, avg_fitness, selection_method, survival_method):
    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness, label="Average Best-So-Far Fitness")
    plt.plot(avg_fitness, label="Average Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness (Lower is better)")
    plt.legend()
    plt.title(f"JSSP Optimization ({selection_method} + {survival_method})")
    plt.show()

if __name__ == "__main__":
    selection_methods = ["FPS", "BinaryTournament", "Truncation", "Random"]
    survival_methods = ["Random", "Truncation", "RBS"]

    for sel in selection_methods:
        for surv in survival_methods:
            print(f"\nRunning for {sel} + {surv}")
            evolutionary_algorithm("la19", pop_size=100, generations=50, iterations=10, selection_method=sel, survival_method=surv)
