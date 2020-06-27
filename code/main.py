import torch

from gradientOptimization.goptimization import optimizationMethods, optimizationFunctionFactory

torch.manual_seed(0)
gopt = optimizationMethods(optimizationFunctionFactory.create("ackley"))

print("regular gradient descent")
x_best, y_best = gopt.gradient_descent(0.01, 1, 1000)

print("x_best ", x_best)
print("y_best ", y_best)

print("perturbed gradient descent")

torch.manual_seed(0)
x_best, y_best = gopt.perturbed_gradient_descent(0.01, 1, 1000, 1)

print("x_best ", x_best)
print("y_best ", y_best)


print("modified perturbed gradient descent")

torch.manual_seed(0)
x_best, y_best = gopt.modified_gradient_descent(0.01, 1, 1000, 1, .01)

print("x_best ", x_best)
print("y_best ", y_best)