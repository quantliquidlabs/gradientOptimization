import torch
import math

from collections import OrderedDict
from enum import Enum

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

    @classmethod
    def transform_linearly(cls, alpha, beta, functor):
        def f(x):  return alpha*functor(x) + beta
        return optimizationFuntion(functor.dim, f)



class exploraitonObject:
    class collectionType(Enum):
        NONE = 0,
        ALL = 1,
        IMPROVEMENTS = 2

    def __init__(self, verbose, type_ = collectionType.IMPROVEMENTS ):
        self.values = OrderedDict()
        self.type = type_
        self.verbose = self.__callVerbose__ if verbose else self.__callNoVerbose__

        self.addP1 = self.__add__ if self.type == self.collectionType.ALL else self.__nothing__
        self.addP2 = self.__add__ if self.type == self.collectionType.IMPROVEMENTS else self.__nothing__

    def start(self, x, y):
        self.y_best = y
        self.x_best = x

    def __callVerbose__(self, s, x, y):
        print("iteration ", s, ", x ", x, ", y ", y)

    def __add__(self, s, x, y):
        self.values[s] = ( x, y )
        self.verbose(s,x,y)

    def __nothing__(self,s, x, y): return

    def __call__(self, s, x, y):
        self.addP1(s, x, y)
        if( y.item() < self.y_best ):
            self.start(x.tolist(), y.item())
            self.addP2(s, self.x_best, self.y_best)

class optimizationMethods:
    class env:
        def __init__(self, kwargs):
            for k,v in kwargs.items():
                setattr(self, k, v)

    class updateFunctor:
        def __init__(self, locals_to_store, objs_attr, func):
            self.locals = locals_to_store
            self.objs = objs_attr
            self.func   = func

        def set_env(self, locals):
            toCache = {y: locals.get(y, None) for y in self.locals}
            toCache.update( {a[1]: getattr( locals.get(a[0], None), a[1], None ) for a in self.objs } )
            self.env = optimizationMethods.env(toCache)

        def __call__(self, x):
            return self.func(x, self.env)

    def __init__(self, function, dim = 2):
        self.func = function if type(function) == optimizationFuntion else \
            optimizationFunctionFactory.create(function, dim)

    def __generalDescent__(self,
                           epochs,
                           steps,
                           update_rule,
                           save_evolution,
                           initial_rule_per_eppoch = None):

        save_evolution.start([], float("inf"))
        if not initial_rule_per_eppoch: initial_rule_per_eppoch = self.__defaultInitial__()
        for _ in range(epochs):
            x = initial_rule_per_eppoch(self.func.dim)
            for s in range(steps):
                y = self.func(x)
                y.backward()
                save_evolution(s,x,y)
                update_rule.set_env(locals())
                x = update_rule(x)

        return save_evolution

    def __defaultInitial__(self, x0 = 3.9, y0 = 4.0):
        def f(d):
            x = torch.FloatTensor(d).uniform_(x0, y0)
            x.requires_grad = True
            return x
        return f

    def gradient_descent(self,
                         lr,
                         epochs,
                         steps,
                         verbose=True,
                         expType=exploraitonObject.collectionType.IMPROVEMENTS,
                         initial_rule = None):
        expObj = exploraitonObject(verbose, expType)
        def update(x, env ):
            with torch.no_grad():
                x -= lr*x.grad
                x.grad.zero_()
            return x

        return self.__generalDescent__(epochs,
                                       steps,
                                       self.updateFunctor([], [], update),
                                       expObj,
                                       initial_rule)

    def perturbed_gradient_descent(self,
                                   lr,
                                   epochs,
                                   steps,
                                   perturbation_ratio,
                                   verbose = True,
                                   expType = exploraitonObject.collectionType.IMPROVEMENTS,
                                   initial_rule = None):
        expObj = exploraitonObject(verbose, expType)
        std = perturbation_ratio / math.sqrt(self.func.dim) * torch.ones(self.func.dim)
        def update(x, env = self.env({}) ):
            with torch.no_grad():
                x -= lr * torch.normal(x.grad, std)
                x.grad.zero_()
            return x

        return self.__generalDescent__(epochs,
                                       steps,
                                       self.updateFunctor([], [], update),
                                       expObj,
                                       initial_rule)

    def modified_gradient_descent(self,
                                  lr,
                                  epochs,
                                  steps,
                                  perturbation_ratio,
                                  gamma,
                                  verbose = True,
                                  expType = exploraitonObject.collectionType.IMPROVEMENTS,
                                  initial_rule = None):
        expObj = exploraitonObject(verbose, expType)
        std = perturbation_ratio / math.sqrt(self.func.dim) * torch.ones(self.func.dim)
        def update(x, env):
            with torch.no_grad():
                x -= lr * torch.normal(math.exp(-gamma * (env.y - env.y_best)) * x.grad, std)
                x.grad.zero_()
            return x

        return self.__generalDescent__(epochs,
                                       steps,
                                       self.updateFunctor(['y'], [ ['save_evolution','y_best'] ], update),
                                       expObj,
                                       initial_rule)