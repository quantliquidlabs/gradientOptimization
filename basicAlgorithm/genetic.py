import random


class GeneCoder:
    def __init__(self, lower_bound, upper_bound, gene_length):
        self.param_width = upper_bound - lower_bound
        self.lower_bound = lower_bound

        self.gene_length = gene_length
        self.max_gene = 2 ** gene_length - 1

    def code(self, value):
        gene = int(self.max_gene * (value - self.lower_bound) / self.param_width)
        gene = min(max(0, gene), self.max_gene)
        return gene

    def decode(self, gene):
        param = gene * self.param_width / self.max_gene + self.lower_bound
        return param


class ChromosomeCoder:
    def __init__(self, name_to_coder):
        self.name_to_coder = name_to_coder
        self.chromosome_length = 0
        for coder in name_to_coder.values():
            self.chromosome_length += coder.gene_length

    def code(self, name_to_value):
        chromosome = int(0)
        for name in sorted(self.name_to_coder):
            coder = self.name_to_coder[name]
            value = name_to_value[name]
            gene = coder.code(value)
            chromosome = chromosome << coder.gene_length
            chromosome += gene

        return chromosome

    def decode(self, chromosome):
        name_to_param = dict()
        for name in sorted(self.name_to_coder, reverse=True):
            coder = self.name_to_coder[name]
            residue = 1 << coder.gene_length
            gene = chromosome % residue
            chromosome = chromosome >> coder.gene_length
            name_to_param[name] = coder.decode(gene)

        return name_to_param


def selection(chromosome_to_fitness):
    population = [i for i in chromosome_to_fitness.keys()]
    fitness = [i for i in chromosome_to_fitness.values()]

    chosen = random.choices(population, weights=fitness)
    return chosen[0]


def crossover(parent_1, parent_2, length):
    selector = random.randrange(1 << length)
    offspring_1 = (parent_1 & ~selector) | (parent_2 & selector)
    offspring_2 = (parent_1 & selector) | (parent_2 & ~selector)
    return offspring_1, offspring_2


def mutation(gene, length, mutation_rate):
    selector = 0
    for _ in range(length):
        selector = selector << 1
        flip = random.choices([0, 1], weights=[1-mutation_rate, mutation_rate])
        selector += flip[0]

    mutated = gene ^ selector
    return mutated


class GeneticAlgorithm:
    def __init__(self, chromosome_coder, fitness_function, initial_population, mutation_rate, batch_size):
        self.chromosome_coder = chromosome_coder
        self.fitness_function = fitness_function
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size

        self.chromosome_length = chromosome_coder.chromosome_length

        self.chromosome_to_fitness = {}

        for chromosome in initial_population:
            self.get_fitness(chromosome)

    def get_fitness(self, chromosome):
        if chromosome in self.chromosome_to_fitness:
            return self.chromosome_to_fitness[chromosome]
        else:
            fitness = self.fitness_function(self.chromosome_coder.decode(chromosome))
            self.chromosome_to_fitness[chromosome] = fitness
            return fitness

    def update_population(self):
        for _ in range(self.batch_size):
            parent_1 = selection(self.chromosome_to_fitness)
            parent_2 = selection(self.chromosome_to_fitness)

            offspring_1, offspring_2 = crossover(parent_1, parent_2, self.chromosome_length)

            offspring_1 = mutation(offspring_1, self.chromosome_length, self.mutation_rate)
            offspring_2 = mutation(offspring_2, self.chromosome_length, self.mutation_rate)

            self.get_fitness(offspring_1)
            self.get_fitness(offspring_2)
