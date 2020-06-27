import torch

from gradientOptimization.goptimization import optimizationMethods, optimizationFunctionFactory

torch.manual_seed(0)
gopt = optimizationMethods(optimizationFunctionFactory.create("ackley"))

print("regular gradient descent")
res = gopt.gradient_descent(0.01, 1, 1000)

print("x_best ", res.x_best)
print("y_best ", res.y_best)

print("perturbed gradient descent")

torch.manual_seed(0)
res = gopt.perturbed_gradient_descent(0.01, 1, 1000, 1)

print("x_best ", res.x_best)
print("y_best ", res.y_best)


print("modified perturbed gradient descent")

torch.manual_seed(0)
res = gopt.modified_gradient_descent(0.01, 1, 1000, 1, .000001)

print("x_best ", res.x_best)
print("y_best ", res.y_best)