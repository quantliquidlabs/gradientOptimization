import torch

import torch.nn.functional as F


# Testing function
def train(network, optimizer, data_loader):
    train_losses = []
    network.train()
    optimizer.best_loss = float("inf")
    for batch_idx, (data, target) in enumerate(data_loader):
        def closure():
            optimizer.zero_grad()
            output = network(data)
            current_loss = F.nll_loss(output, target)
            current_loss.backward()
            return current_loss

        loss = optimizer.step(closure)

        train_losses.append(loss.item())

    return train_losses


# Test function
def test(network, data_loader):

    network.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()

    test_loss /= len(data_loader.dataset)

    return test_loss