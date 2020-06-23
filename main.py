import torch

import goptimization as gop

'''
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-1, 1, 0.005)
Y = np.arange(-1, 1, 0.005)
X, Y = np.meshgrid(X, Y)
Z = 20 + (5.12 * X) ** 2 - 10 * np.cos(2 * np.pi * (5.12 * X)) + (5.12 * Y) ** 2 - 10 * np.cos(2 * np.pi * (5.12 * Y))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z)

plt.show()
'''

print("regular gradient descent")
torch.manual_seed(0)
x_best, y_best = gop.gradient_descent(gop.ackley_fun, 2, 0.0001, 1, 1000)

print("x_best ", x_best)
print("y_best ", y_best)

print("perturbed gradient descent")

torch.manual_seed(0)
x_best, y_best = gop.perturbed_gradient_descent(gop.ackley_fun, 2, 0.0001, 1, 1000, 1000)

print("x_best ", x_best)
print("y_best ", y_best)


print("modified perturbed gradient descent")

torch.manual_seed(0)
x_best, y_best = gop.modified_gradient_descent(gop.ackley_fun, 2, 0.0001, 1, 1000, 1000, 10)

print("x_best ", x_best)
print("y_best ", y_best)