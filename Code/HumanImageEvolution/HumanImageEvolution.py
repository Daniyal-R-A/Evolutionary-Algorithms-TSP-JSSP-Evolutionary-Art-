from types import FunctionType
from typing import List
from tqdm import tqdm
from abc import ABC, abstractmethod
import random
import copy
import numpy as np
import threading
from PIL import Image, ImageDraw

class Individual():
  def __init__(self, genome, fitness):
    self.genome = genome
    self.fitness = fitness


  @abstractmethod
  def mutate(self) -> None:
    pass


class EvolutionaryAlgorithm():
  def __init__(self, 
               initial_population_function: FunctionType,
               parent_selection_function: str,
               survivor_selection_function: str,
               cross_over_function: FunctionType,
               population_size: int = 100,
               mutation_rate: float = 0.5,
               num_offsprings: int = 10):
    selection_functions_string_map = {'truncation': self.truncation_selection,
                                      'random': self.random_selection,
                                      'binary': self.binary_tournament_selection,
                                      'rank': self.rank_selection,
                                      'fitness': self.fitness_proportional_selection}
    self.initial_population_function: FunctionType = initial_population_function
    self.population: List[Individual] = self.initial_population_function(population_size)
    self.population_size: int = population_size
    self.mutation_rate: float = mutation_rate
    self.parent_selection_function: FunctionType = selection_functions_string_map[parent_selection_function]
    self.survivor_selection_function: FunctionType = selection_functions_string_map[survivor_selection_function]
    self.cross_over_function: FunctionType = cross_over_function
    self.num_offsprings: int = num_offsprings


  ## selection functions
  def random_selection(self, num_selections: int) -> List[Individual]:
    survivors = []
    for i in range(num_selections):
      random_int = random.randint(0, len(self.population)-1)
      survivors.append(self.population[random_int])
    return survivors


  def truncation_selection(self, num_selections: int) -> List[Individual]:
    result = []
    result = copy.deepcopy(self.population)
    result.sort(key=lambda k : k.fitness)
    return result[:num_selections]
  

  def binary_tournament_selection(self, num_selections: int) -> List[Individual]:
    result= []
    for i in range(num_selections):
        ind1, ind2 = random.sample(self.population, 2)
        selected = ind1 if ind1.fitness < ind2.fitness else ind2
        result.append(selected)
    return result


  def rank_selection(self, num_selections: int) -> List[Individual]:
    self.population.sort(key=lambda individual: individual.fitness, reverse=True)
    ranks = np.arange(1, len(self.population) + 1)
    total_rank = np.sum(ranks)
    selection_probs = ranks / total_rank
    selected_indices = np.random.choice(range(len(self.population)), size=num_selections, replace=True, p=selection_probs)
    return [self.population[i] for i in selected_indices]
  

  def fitness_proportional_selection(self, num_selections: int) -> List[Individual]:
    total_fitness = sum(1/individual.fitness for individual in self.population)
    selection_probs = [(1/individual.fitness) / total_fitness for individual in self.population]
    selected_indices = np.random.choice(range(len(self.population)), size=num_selections, replace=True, p=selection_probs)
    return [self.population[i] for i in selected_indices]


  def get_average_and_best_individual(self) -> (Individual, float):
    best_individual = self.population[0]
    cumulative_fitness = 0
    for individual in self.population:
      if(individual.fitness < best_individual.fitness):
        best_individual = individual
      cumulative_fitness += individual.fitness
    average_fitness = cumulative_fitness/len(self.population)
    return best_individual, average_fitness


  def get_total_fitness(self) -> float:
    total_fitness = 0
    for individual in self.population:
      total_fitness += individual.fitness
    return total_fitness


  def run_generation(self) -> None:
    parents = self.parent_selection_function(self.num_offsprings)

    # creating offspring
    for k in range(0, self.num_offsprings-1, 2):
      offspring1, offspring2 = self.cross_over_function(parents[k], parents[k+1])
      rand_num1, rand_num2 = random.randint(0,100)/100, random.randint(0,100)/100
      if rand_num1 <= self.mutation_rate:
        offspring1.mutate()
      if rand_num2 <= self.mutation_rate:
        offspring2.mutate()
      self.population.extend([offspring1, offspring2])

    self.population = self.survivor_selection_function(self.population_size)


  def process_offspring_range(self, start, end, parents, lock):
    for k in range(start, end, 2):
        offspring1, offspring2 = self.cross_over_function(parents[k], parents[k + 1])
        rand_num1, rand_num2 = random.randint(0, 100) / 100, random.randint(0, 100) / 100
        if rand_num1 <= self.mutation_rate:
            offspring1.mutate()
        if rand_num2 <= self.mutation_rate:
            offspring2.mutate()
        with lock:
            self.population.extend([offspring1, offspring2])


  def run_generation_threaded(self) -> None:
    parents = self.parent_selection_function(self.num_offsprings)
    threads = []
    lock = threading.Lock()
    num_threads = 4  # Number of threads you want to create
    chunk_size = (self.num_offsprings - 1) // num_threads

    for i in range(num_threads):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, self.num_offsprings)
        thread = threading.Thread(target=self.process_offspring_range, args=(start, end, parents, lock))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    self.population = self.survivor_selection_function(self.population_size)
        

  @abstractmethod
  def run():
    pass
    

# Set up the screen
width, height = 1200, 1200

max_polygons_per_image = 50
min_polygons_per_image = 5

# Load the reference image
reference_image = np.array(Image.open('HumanImageEvolution/monalisa.png'))


def draw_polygon(draw, color, vertices) -> None:
    draw.polygon(vertices, fill=color)


def generate_random_polygon():
    num_vertices = random.randint(3, 10)  # Random number of vertices (3 to 6)
    vertices = [(random.randint(0, width), random.randint(0, height)) for _ in range(num_vertices)]
    # color = color_pallete[np.random.randint(0, 5)]  # Random RGB color
    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), random.randint(10,60))  # Random RGB color
    return {'vertices': vertices, 'color': color}


def image_difference(genome):
    image = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    # Draw each polygon
    for polygon in genome:
        draw_polygon(draw, polygon['color'], polygon['vertices'])
    # Calculate the absolute pixel-wise difference
    diff = np.abs(reference_image - np.array(image))
    # Calculate the mean difference as the fitness score
    image_difference = np.mean(diff)
    return image_difference

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))

class PolygonImage(Individual):
    def __init__(self, genome):
        fitness = image_difference(genome)
        super().__init__(genome, fitness)

        # These values have been adapted from preliminary implementation by oysteinkrog
        self.mutation_rates = {"add_polygon": 0.0015,
                               "remove_polygon": 0.00067,
                               "move_polygon": 0.0015,
                               "add_point": 0.00067,
                               "remove_point": 0.00067,
                               "large_point_change": 0.00067,
                               "medium_point_change": 0.00067,
                               "small_point_change": 0.00067,
                               "mutate_red": 0.00067,
                               "mutate_green": 0.00067,
                               "mutate_blue": 0.00067,
                               "mutate_alpha": 0.00067}
        self.min_points_per_polygon = 3
        self.max_points_per_polygon = 10
        self.mutation_ranges = {
                                "min_points_per_polygon": 3,
                                "max_points_per_polygon": 10,
                                "medium_point_change": 20,
                                "small_point_change": 3,
                                }
    
    def add_polygon(self) -> None:
        if(len(self.genome) < max_polygons_per_image):
            index = random.randint(0, len(self.genome)-1)
            vertices = []
            for i in range(self.mutation_ranges["min_points_per_polygon"]):
                vertices.append((random.randint(0, width), random.randint(0, height)))
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), random.randint(10,60))  # Random RGB color
            self.genome.insert(index, {'vertices': vertices, 'color': color})


    def remove_polygon(self) -> None:
        if(len(self.genome) > min_polygons_per_image):
            index = random.randint(0, len(self.genome)-1)
            del self.genome[index]
            
    def move_polygon(self) -> None:
        index1 = random.randint(0, len(self.genome)-1)
        index2 = random.randint(0, len(self.genome)-1)
        self.genome[index1], self.genome[index2] = self.genome[index2], self.genome[index1]

    def add_point(self, vertices) -> List:
        if(len(vertices) > self.max_points_per_polygon):
            return vertices
        index = random.randint(0, len(vertices)-2)
        prevX, prevY = vertices[index]
        nextX, nextY = vertices[index+1]
        newX, newY = (prevX+nextX)/2, (prevY+nextY)/2
        vertices.append((newX, newY))
        return vertices

    def remove_point(self, vertices) -> List:
        if(len(vertices) < self.min_points_per_polygon):
            return vertices
        index = random.randint(0, len(vertices)-1)
        del vertices[index]
        return vertices


    def mutate_color(self, color) -> tuple:
        color = list(color)
        if(np.random.uniform() <= self.mutation_rates["mutate_red"]):
            color[0] = random.randint(0, 255)
        if(np.random.uniform() <= self.mutation_rates["mutate_red"]):
            color[1] = random.randint(0, 255)
        if(np.random.uniform() <= self.mutation_rates["mutate_red"]):
            color[2] = random.randint(0, 255)
        if(np.random.uniform() <= self.mutation_rates["mutate_red"]):
            color[3] = random.randint(10, 60)
        return tuple(color)
        

    def mutate_points(self, vertices) -> list:
        for i in range(len(vertices)):
            if(np.random.uniform() <= self.mutation_rates["large_point_change"]):
                vertices[i] = (random.randint(0, width), random.randint(0, height))
            if(np.random.uniform() <= self.mutation_rates["medium_point_change"]):
                x = clamp(vertices[i][0] + np.random.uniform(self.mutation_ranges["medium_point_change"]), 0, width)
                y = clamp(vertices[i][1] + np.random.uniform(self.mutation_ranges["medium_point_change"]), 0, height)
                vertices[i] = (x, y)
            if(np.random.uniform() <= self.mutation_rates["small_point_change"]):
                x = clamp(vertices[i][0] + np.random.uniform(self.mutation_ranges["small_point_change"]), 0, width)
                y = clamp(vertices[i][1] + np.random.uniform(self.mutation_ranges["small_point_change"]), 0, height)
                vertices[i] = (x, y)

        return vertices


    def mutate_polygons(self) -> None:
        for i in range(len(self.genome)):
            if(np.random.uniform() <= self.mutation_rates["add_point"]):
                self.genome[i]["vertices"] = self.add_point(self.genome[i]["vertices"])
            if(np.random.uniform() <= self.mutation_rates["remove_point"]):
                self.genome[i]["vertices"] = self.remove_point(self.genome[i]["vertices"])

            self.genome[i]["color"] = self.mutate_color(self.genome[i]["color"])
            self.genome[i]["vertices"] = self.mutate_points(self.genome[i]["vertices"])



    def mutate(self) -> None:
        # Add polygon
        if(np.random.uniform() <= self.mutation_rates["add_polygon"]):
            self.add_polygon()
        if(np.random.uniform() <= self.mutation_rates["remove_polygon"]):
            self.remove_polygon()
        if(np.random.uniform() <= self.mutation_rates["move_polygon"]):
            self.move_polygon()

        self.mutate_polygons()
        
        self.fitness = image_difference(self.genome)



    def save(self, image_name):
        image = Image.new("RGB", (width, height), color=(0,0,0))
        draw = ImageDraw.Draw(image, "RGBA")
        # Draw each polygon
        for polygon in self.genome:
            draw_polygon(draw, polygon['color'], polygon['vertices'])
        image.save(image_name)


def random_polygon_combinations(population_size: int) -> List[PolygonImage]:
    population = []
    for i in range(population_size):
        genome = [generate_random_polygon() for _ in range(min_polygons_per_image)]
        population.append(PolygonImage(genome))
    return population


def random_length_crossover(parent1: PolygonImage, parent2: PolygonImage) -> tuple:
    length1 = len(parent1.genome)
    length2 = len(parent2.genome)
    max_length = max(length1, length2)
    min_length = min(length1, length2)

    start = random.randint(0, int(min_length-3))
    end = random.randint(start, int(min_length-2))

    offspring1 = [None] * max_length
    offspring2 = [None] * max_length

    offspring1[start:end+1] = parent1.genome[start:end+1]
    offspring2[start:end+1] = parent2.genome[start:end+1]

    pointer = end + 1
    parent1_pointer = end + 1
    parent2_pointer = end + 1

    while None in offspring1:
        #if parent2.genome[parent2_pointer] not in offspring1:
        offspring1[pointer % max_length] = parent2.genome[parent2_pointer]
        pointer += 1
        parent2_pointer = (parent2_pointer + 1) % length2

    pointer = 0

    while None in offspring2:
        #if parent1.genome[parent1_pointer] not in offspring2:
        offspring2[pointer % max_length] = parent1.genome[parent1_pointer]
        pointer += 1
        parent1_pointer = (parent1_pointer + 1) % length1

    offspring1 = PolygonImage(offspring1)
    offspring2 = PolygonImage(offspring2)

    return offspring1, offspring2


class MonaLisa_EvolutionaryAlgorithm(EvolutionaryAlgorithm):
    def run(self, num_iterations: int=10, num_generations: int=10000):
      for j in range(num_iterations):
        for i in tqdm(range(num_generations), desc='Iteration '+str(j+1)):
          self.run_generation_threaded()
          if(i % 50 == 0):
            best_individual, average_fitness = self.get_average_and_best_individual()
            print("\nAverage fitness: ", average_fitness, ", Best value: ", best_individual.fitness)
            best_individual.save("data/fake_monalisa_"+str(j)+"_"+str(i)+".png")

        self.population = self.initial_population_function(self.population_size)
        

monalisa = MonaLisa_EvolutionaryAlgorithm(
    initial_population_function = random_polygon_combinations,
    parent_selection_function = 'truncation',
    survivor_selection_function = 'truncation',
    cross_over_function = random_length_crossover,
    population_size = 100,
    mutation_rate = 0.5,
    num_offsprings=50
)
monalisa.run(num_generations=10000)