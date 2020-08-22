import torch
import math
import random


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


def paraboloid_fun(x):
    return torch.dot(x, x)


# Optimization algorithms
def gradient_descent(fun, x0, steps, lr):
    x_best = []
    y_best = float("inf")

    x = x0.detach().clone()
    x.requires_grad = True

    x_list = []
    y_list = []

    for _ in range(steps):
        y = fun(x)
        y.backward()

        x_list.append(x.tolist())
        y_list.append(y.item())

        if y_best > y.item():
            x_best = x.tolist()
            y_best = y.item()

        with torch.no_grad():
            x -= lr * x.grad
            x.grad.zero_()

    return x_best, y_best, x_list, y_list


def perturbed_gradient_descent(fun, x0, steps, lr, perturbation_ratio):
    x_best = []
    y_best = float("inf")

    dim = sum(x0.shape)

    std = perturbation_ratio / math.sqrt(dim) * torch.ones(dim)

    x = x0.detach().clone()
    x.requires_grad = True

    x_list = []
    y_list = []

    for _ in range(steps):
        y = fun(x)
        y.backward()

        x_list.append(x.tolist())
        y_list.append(y.item())

        if y_best > y.item():
            x_best = x.tolist()
            y_best = y.item()

        with torch.no_grad():
            x -= lr * torch.normal(x.grad, std)
            x.grad.zero_()

    return x_best, y_best, x_list, y_list


def modified_gradient_descent(fun, x0, steps, lr, perturbation_ratio, gamma):
    x_best = []
    y_best = float("inf")

    dim = sum(x0.shape)

    std = perturbation_ratio / math.sqrt(dim) * torch.ones(dim)

    x = x0.detach().clone()
    x.requires_grad = True

    x_list = []
    y_list = []

    for _ in range(steps):
        y = fun(x)
        y.backward()

        x_list.append(x.tolist())
        y_list.append(y.item())

        if y_best > y.item():
            x_best = x.tolist()
            y_best = y.item()

        with torch.no_grad():
            x -= lr * torch.normal(math.exp(-gamma * (y - y_best)) * x.grad, std)
            x.grad.zero_()

    return x_best, y_best, x_list, y_list


def gradient_descent_double_gamma(fun, x0, steps, lr, perturbation_ratio, gamma_1, gamma_2, p_gamma):
    x_best = []
    y_best = float("inf")

    dim = sum(x0.shape)

    std = perturbation_ratio / math.sqrt(dim) * torch.ones(dim)

    x = x0.detach().clone()
    x.requires_grad = True

    x_list = []
    y_list = []

    for _ in range(steps):
        y = fun(x)
        y.backward()

        x_list.append(x.tolist())
        y_list.append(y.item())

        if y_best > y.item():
            x_best = x.tolist()
            y_best = y.item()

        with torch.no_grad():
            gamma = gamma_1 if random.random() < p_gamma else gamma_2
            x -= lr * torch.normal(math.exp(-gamma * (y - y_best)) * x.grad, std)
            x.grad.zero_()

    return x_best, y_best, x_list, y_list
