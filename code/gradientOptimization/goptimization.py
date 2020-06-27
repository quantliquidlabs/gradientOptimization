import torch
import math

### Optimization Functions
class optimizationFuntion:
    def __init__(self, dim, functor):
        self.dim, self.functor = dim, functor

    def __call__(self, x):
        return self.functor(x)

class optimizationFunctionFactory:
    @staticmethod
    def ratrigin_fun(x):
        n = len(x)
        out = 10 * n + sum([(5.12 * x_i) ** 2 - 10 * torch.cos(2 * math.pi * (5.12 * x_i)) for x_i in x])
        return out

    @staticmethod
    def ackley_fun(x):
        out = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * sum([(5 * x_i) ** 2 for x_i in x])))
        out -= torch.exp(0.5 * sum([torch.cos(2 * math.pi * 5 * x_i) for x_i in x]))
        out += math.e + 20
        return out

    @classmethod
    def create(cls, type, dim = 2):
        if type == "ackley": return optimizationFuntion( dim, cls.ackley_fun )
        elif type == "ratrigin": return optimizationFuntion( dim, cls.ratrigin_fun )
        raise RuntimeError("Not recognized funciton type")

class optimizationMethods:
    class env:
        def __init__(self, kwargs):
            for k,v in kwargs.items():
                setattr(self, k, v)

    class updateFunctor:
        def __init__(self, locals_to_store, func):
            self.locals, self.func = locals_to_store, func

        def set_env(self, locals):
            self.env = optimizationMethods.env({y:locals.get(y, None) for y in self.locals})

        def __call__(self, x):
            return self.func(x, self.env)

    def __init__(self, function, dim = 2):
        self.func = function if type(function) == optimizationFuntion else \
            optimizationFunctionFactory.create(function, dim)

    def __generalDescent__(self, epochs, steps, update_rule, initial_rule_per_eppoch = None):
        x_best = []
        y_best = float("inf")

        if not initial_rule_per_eppoch: initial_rule_per_eppoch = self.__defaultInitial__()

        for _ in range(epochs):
            x = initial_rule_per_eppoch(self.func.dim)
            for s in range(steps):
                y = self.func(x)
                y.backward()

                if y_best > y.item():
                    x_best = x.tolist()
                    y_best = y.item()
                    print("iteration ", s, " best y", y_best)

                update_rule.set_env(locals())
                x = update_rule(x)

        return x_best, y_best

    def __defaultInitial__(self):
        def f(d):
            x = torch.FloatTensor(d).uniform_(3.9, 4)
            x.requires_grad = True
            return x
        return f

    def gradient_descent(self, lr, epochs, steps, initial_rule = None):
        def update(x, env ):
            with torch.no_grad():
                x -= lr*x.grad
                x.grad.zero_()
            return x

        return self.__generalDescent__(epochs, steps, self.updateFunctor([], update), initial_rule)

    def perturbed_gradient_descent(self, lr, epochs, steps, perturbation_ratio, initial_rule = None):
        std = perturbation_ratio / math.sqrt(self.func.dim) * torch.ones(self.func.dim)
        def update(x, env = self.env({}) ):
            with torch.no_grad():
                x -= lr * torch.normal(x.grad, std)
                x.grad.zero_()
            return x

        return self.__generalDescent__(epochs, steps, self.updateFunctor([], update), initial_rule)

    def modified_gradient_descent(self, lr, epochs, steps, perturbation_ratio, gamma, initial_rule = None):
        std = perturbation_ratio / math.sqrt(self.func.dim) * torch.ones(self.func.dim)
        def update(x, env):
            with torch.no_grad():
                x -= lr * torch.normal(math.exp(-gamma * (env.y - env.y_best)) * x.grad, std)
                x.grad.zero_()
            return x

        return self.__generalDescent__(epochs, steps, self.updateFunctor(['y','y_best'], update), initial_rule)