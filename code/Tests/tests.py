from unittest import TestCase, main

import torch

from gradientOptimization.goptimization import optimizationMethods, optimizationFunctionFactory


class testBasicPerformace(TestCase):
    def setUp(self) -> None:
        self.gopt = optimizationMethods(optimizationFunctionFactory.create("ackley"))
        self.gamma = 0.01
        self.epochs = 1
        self.steps = 100
        self.sigma = 1

    def test_zero_gamma(self):
        torch.manual_seed(0)
        perturbed = self.gopt.perturbed_gradient_descent(self.gamma, self.epochs, self.steps, self.sigma, verbose=False)

        torch.manual_seed(0)
        modified = self.gopt.modified_gradient_descent(self.gamma, self.epochs, self.steps, self.sigma, 0.0,
                                                       verbose=False)

        self.assertAlmostEqual(perturbed.x_best[0], modified.x_best[0], 5,
                               "Execution of modified vs perturbed diverge in x direciton")
        self.assertAlmostEqual(perturbed.x_best[1], modified.x_best[1], 5,
                               "Execution of modified vs perturbed diverge in y direciton")
        self.assertAlmostEqual(perturbed.y_best, modified.y_best, 5,
                               "Execution of modified vs perturbed diverge in z direciton")


# torch.manual_seed(0)
# gopt = optimizationMethods(optimizationFunctionFactory.create("ackley"))

# print("regular gradient descent")
# res = gopt.gradient_descent(0.001, 1, 1000)

# print("x_best ", res.x_best)
# print("y_best ", res.y_best)

# print("perturbed gradient descent")

# torch.manual_seed(0)
# res = gopt.perturbed_gradient_descent(0.01, 1, 1000, 1)

# print("x_best ", res.x_best)
# print("y_best ", res.y_best)

# print("modified perturbed gradient descent")

# torch.manual_seed(0)
# res = gopt.modified_gradient_descent(0.01, 1, 1000, 1, 0.0001)

# print("x_best ", res.x_best)
# print("y_best ", res.y_best)

if __name__ == '__main__':
    main()
