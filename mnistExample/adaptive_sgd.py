import copy

import math

import random

import torch


# Creating the new optimizer
class AdaptiveSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, pr=0, gamma_1=0, gamma_2=0, p_gamma=0):
        if lr <= 0:
            raise ValueError("Invalid learning rate: lr = {} must be positive".format(lr))
        if pr < 0:
            raise ValueError("Invalid perturbation ratio: lr = {} must be positive".format(pr))

        defaults = dict(lr=lr, pr=pr, gamma_1=gamma_1, gamma_2=gamma_2, p_gamma=p_gamma)

        super(AdaptiveSGD, self).__init__(params, defaults)

        self.gamma = gamma_1

        self.best_loss = float("inf")

    def __setstate__(self, state):
        super(AdaptiveSGD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is None:
            raise ValueError("AdaptiveSGD requires a closure function.")

        with torch.enable_grad():
            loss = closure()

        if loss < self.best_loss:
            self.best_loss = loss

        for group in self.param_groups:
            lr = group['lr']
            pr = group['pr']
            gamma_1 = group['gamma_1']
            gamma_2 = group['gamma_2']
            p_gamma = group['p_gamma']

            self.gamma = gamma_1 if random.random() < p_gamma else gamma_2

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                dim = sum(p.shape)
                std = pr / math.sqrt(dim) * torch.ones(p.shape)

                p.add_(torch.normal(d_p, std), alpha=-lr * math.exp(-self.gamma * (loss - self.best_loss)))

        return loss

