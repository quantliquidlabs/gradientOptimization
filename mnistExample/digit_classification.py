import adaptive_sgd

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# Learning parameters
n_epochs = 1
batch_size_train = 64
batch_size_test = 1000

learning_rate = 0.01
perturbation_rate = 0
gamma_1 = 0
gamma_2 = 0.01
p_gamma = 0.5

log_interval = 1

# initialization parameters
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Loading dataset
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

# getting data
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

# # plotting some examples
# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Ground Truth: {}".format(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])
#
# plt.show()


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


# Initialize network and optimizer
network = Net()

optimizer = adaptive_sgd.AdaptiveSGD(network.parameters(),
                                     lr=learning_rate, pr=perturbation_rate,
                                     gamma_1=gamma_1, gamma_2=gamma_2, p_gamma=p_gamma)

# Training
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


# Testing function
def train(epoch):
    network.train()
    optimizer.best_loss = float("inf")
    for batch_idx, (data, target) in enumerate(train_loader):

        def closure():
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        train_losses.append(loss.item())

        train_counter.append(
            (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Gamma: {}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), loss.item(), optimizer.gamma))


# Test function
def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    print('current best loss on data {:.4f}'.format(optimizer.best_loss.item()))
    test()


torch.save(network.state_dict(), './results/model.pth')
torch.save(optimizer.state_dict(), './results/optimizer.pth')

# Plot training curve
plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()

# # check examples
# with torch.no_grad():
#     output = network(example_data)
#
# plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Prediction: {}".format(
#         output.data.max(1, keepdim=True)[1][i].item()))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

# # Train from previous check point
# continued_network = Net()
# continued_optimizer = adaptive_sgd.AdaptiveSGD(network.parameters(), lr=learning_rate, pr=perturbation_rate,
#                                                gamma_1=gamma_1, gamma_2=gamma_2, p_gamma=p_gamma)
#
#
# # Loading previous network
# network_state_dict = torch.load('./results/model.pth')
# continued_network.load_state_dict(network_state_dict)
#
# optimizer_state_dict = torch.load('./results/optimizer.pth')
# continued_optimizer.load_state_dict(optimizer_state_dict)
#
# # Training some more
# for i in range(4, 9):
#     test_counter.append(i * len(train_loader.dataset))
#     train(i)
#     test()
#
# # Final performance
# plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# plt.show()
