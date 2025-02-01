import sys
import random

class TSP:
    def __init__(self, dataset, population_size):
        self.dataset = dataset
        self.no_of_nodes = len(dataset)
        self.population_size = population_size
        self.chromosomes = [] # list of chromosomes which have individual cities in random order
        
    
    def generate_chromosomes(self):
        chromosome_length = self.no_of_nodes
        for _ in range(self.population_size):
            # list of indices of cities in random order starting from 0 to chromosome_length - 1
            chromosome = list(range(chromosome_length))
            random.shuffle(chromosome)
            self.chromosomes.append(chromosome)

    def fitness_function(self, chromosome):
        '''
        Calculate distance using the euclidean distance formula
        sqrt((x1 - x2)^2 + (y1 - y2)^2)
        '''
        distance = 0
        for i in range(self.no_of_nodes - 1):
            x1, y1 = self.dataset[chromosome[i]]
            x2, y2 = self.dataset[chromosome[i + 1]]
            distance += ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        
        # distance between the last and the first city
        x1, y1 = self.dataset[chromosome[-1]]
        x2, y2 = self.dataset[chromosome[0]]
        distance += ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return distance

if __name__ == '__main__':
    sys.stdin = open('dataset.txt', 'r', encoding='utf-8')
    sys.stdout = open('output.txt', 'w', encoding='utf-8')
    dataset = [list(map(float, line.split()))[1:] for line in sys.stdin]
    population_size = int(sys.argv[1]) if sys.argv[1] else 30
    tsp = TSP(dataset, population_size)
    tsp.generate_chromosomes()
    # print(tsp.dataset)
    print("Distances:")
    distances = []
    for i, chromosome in enumerate(tsp.chromosomes):
        distances.append(tsp.fitness_function(chromosome))
        # print(i+1 ,tsp.fitness_function(chromosome))

    distances.sort()
    print("Minimum distance:", distances[0:30])