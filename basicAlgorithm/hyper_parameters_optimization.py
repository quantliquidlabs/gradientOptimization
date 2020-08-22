import goptimization as gop
import genetic as gen

import torch
import random

to_optimize = gop.ackley_fun
dim = 2
x0 = 9.9 * torch.ones(dim)
steps = 50


def fitness_function(params):
    torch.manual_seed(0)
    _, y_best, _, _ = gop.gradient_descent_double_gamma(to_optimize,
                                                        x0,
                                                        steps,
                                                        0.083,
                                                        params['pr'],
                                                        params['gamma_1'],
                                                        params['gamma_2'],
                                                        params['p_gamma'])

    return 1 / (y_best + 0.1)


# coder_lr = gen.GeneCoder(0.01, 1, 10)
coder_pr = gen.GeneCoder(0, 0.1, 7)
coder_gamma_1 = gen.GeneCoder(0, 10, 7)
coder_gamma_2 = gen.GeneCoder(0, 10, 7)
coder_p_gamma = gen.GeneCoder(0, 0.25, 5)

name_to_coder = {# 'lr': coder_lr},
                 'pr': coder_pr,
                 'gamma_1': coder_gamma_1,
                 'gamma_2': coder_gamma_2,
                 'p_gamma': coder_p_gamma}

chromosome_coder = gen.ChromosomeCoder(name_to_coder)

batch_size = 20
initial_population = [random.randrange(1 << chromosome_coder.chromosome_length) for _ in range(2 * batch_size)]

mutation_rate = 0.1
genetic_algorithm = gen.GeneticAlgorithm(chromosome_coder,
                                         fitness_function,
                                         initial_population,
                                         mutation_rate,
                                         batch_size)

num_generations = 500
for _ in range(num_generations):
    genetic_algorithm.update_population()

sorted_values = sorted(genetic_algorithm.chromosome_to_fitness.items(), key=lambda x: x[1], reverse=True)

for k in range(10):
    params = chromosome_coder.decode(sorted_values[k][0])
    fitness = sorted_values[k][1]
    print('lr {:.3f} pr {:.3f} gamma_1 {:.3f} gamma_2 {:.3f} p_gamma {:.3f} fitness {:.3f} loss {:.3f}'.format(
        0.083, #params['lr'],
        params['pr'],
        params['gamma_1'],
        params['gamma_2'],
        params['p_gamma'],
        fitness,
        (1 / fitness) - 0.1
    ))
    # print('lr {:.3f} fitness {:.3f} loss {:.3f}'.format( params['lr'], fitness, (1/fitness) - 0.1))

params_best = chromosome_coder.decode(sorted_values[0][0])
fitness_best = sorted_values[0][1]

params_gd = chromosome_coder.decode(sorted_values[0][0])
params_gd['pr'] = 0
params_gd['gamma_1'] = 0
params_gd['gamma_2'] = 0
params_gd['p_gamma'] = 1

fitness_gd = fitness_function(params_gd)

params_sgd = chromosome_coder.decode(sorted_values[0][0])
params_sgd['gamma_1'] = 0
params_sgd['gamma_2'] = 0
params_sgd['p_gamma'] = 1

fitness_sgd = fitness_function(params_sgd)

print('Best error ', (1 / fitness_best) - 0.1)
print('GD error', (1 / fitness_gd) - 0.1)
print('SGD error', (1 / fitness_sgd) - 0.1)
