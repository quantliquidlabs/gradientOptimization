import torch
import math


# Testing Functions
def ratrigin_fun(x):
    n = len(x)
    out = 10 * n + sum([(5.12 * x_i) ** 2 - 10 * torch.cos(2 * math.pi * (5.12 * x_i)) for x_i in x])
    return out


def ackley_fun(x):
    out = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * sum([(5 * x_i) ** 2 for x_i in x])))
    out -= torch.exp(0.5 * sum([torch.cos(2 * math.pi * 5 * x_i) for x_i in x]))
    out += math.e + 20
    return out


# Optimization algorithms
def gradient_descent(fun, dim, lr, epochs, steps):
    x_best = []
    y_best = float("inf")

    for _ in range(epochs):

        x = torch.FloatTensor(dim).uniform_(3.9, 4)
        x.requires_grad = True

        for s in range(steps):
            y = fun(x)
            y.backward()

            if y_best > y.item():
                x_best = x.tolist()
                y_best = y.item()
                print("iteration ", s, " best y", y_best)

            with torch.no_grad():
                x -= lr * x.grad
                x.grad.zero_()

    return x_best, y_best


def perturbed_gradient_descent(fun, dim, lr, epochs, steps, perturbation_ratio):
    x_best = []
    y_best = float("inf")

    std = perturbation_ratio / math.sqrt(dim) * torch.ones(dim)

    for _ in range(epochs):

        x = torch.FloatTensor(dim).uniform_(3.9, 4)
        x.requires_grad = True

        for s in range(steps):
            y = fun(x)
            y.backward()

            if y_best > y.item():
                x_best = x.tolist()
                y_best = y.item()
                print("iteration ", s, " best y", y_best)

            with torch.no_grad():
                x -= lr * torch.normal(x.grad, std)
                x.grad.zero_()

    return x_best, y_best


def modified_gradient_descent(fun, dim, lr, epochs, steps, perturbation_ratio, gamma):
    x_best = []
    y_best = float("inf")

    std = perturbation_ratio / math.sqrt(dim) * torch.ones(dim)

    for _ in range(epochs):

        x = torch.FloatTensor(dim).uniform_(3.9, 4)
        x.requires_grad = True

        for s in range(steps):
            y = fun(x)
            y.backward()

            if y_best > y.item():
                x_best = x.tolist()
                y_best = y.item()
                print("iteration ", s, " best y", y_best)

            with torch.no_grad():
                x -= lr * torch.normal(math.exp(-gamma * (y - y_best)) * x.grad, std)
                x.grad.zero_()

    return x_best, y_best
