import adaptive_sgd

import training

import genetic

import random

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

import pickle

# initialization parameters
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Loading data set
batch_size_train = 64
batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)


# Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def calc_fitness_from_params(params):
    network = Net()

    optimizer = adaptive_sgd.AdaptiveSGD(network.parameters(),
                                         lr=params['lr'],
                                         pr=params['pr'],
                                         gamma_1=params['gamma_1'],
                                         gamma_2=params['gamma_2'],
                                         p_gamma=params['p_gamma'])

    training.train(network, optimizer, train_loader)
    test_loss = training.test(network, test_loader)
    return 1 / test_loss


lr_coder = genetic.GeneCoder(0.01, 0.1, 2)
pr_coder = genetic.GeneCoder(0, 0.1, 4)
g_1_coder = genetic.GeneCoder(0, 0.05, 4)
g_2_coder = genetic.GeneCoder(1, 10, 4)
p_g_coder = genetic.GeneCoder(0, 1, 4)

name_to_coder = {'lr': lr_coder,
                 'pr': pr_coder,
                 'gamma_1': g_1_coder,
                 'gamma_2': g_2_coder,
                 'p_gamma': p_g_coder}

chromosome_coder = genetic.ChromosomeCoder(name_to_coder)

population_size = 10
batch_size = 10

initial_population = [random.randrange(1 << chromosome_coder.chromosome_length) for _ in range(population_size)]
initial_population.append(0)

mutation_rate = 0.1

with open('genetic_algorithm.pkl', 'rb') as in_data:
    genetic_algorithm = pickle.load(in_data)

genetic_algorithm_results = sorted(genetic_algorithm.chromosome_to_fitness.items(),
                                   key=lambda items: items[1],
                                   reverse=True)

for chromosome, fitness in genetic_algorithm_results:
    params = chromosome_coder.decode(chromosome)
    print("lr = {:.3f} pr = {:.3f} gamma_1 = {:.3f} gamma_2 = {:.3f} p_gamma = {:.3f} loss = {:.5f}".format(
        params["lr"],
        params["pr"],
        params["gamma_1"],
        params["gamma_2"],
        params["p_gamma"],
        1 / fitness,
    ))

param_sg = chromosome_coder.decode(genetic_algorithm_results[0][0])

print("best params", param_sg, "error", 1/genetic_algorithm_results[0][1])
param_sg['pr']=0
param_sg['gamma_1']= 0.005
param_sg['gamma_2']= 5
# param_sg['p_gamma']=0
fitness = calc_fitness_from_params(param_sg)
print("error of SGD ", 1 / fitness)
print(param_sg)
