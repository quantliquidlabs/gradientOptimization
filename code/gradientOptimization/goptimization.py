import torch
import math

from enum import Enum
from typing import Callable, List, Tuple
from collections import defaultdict, OrderedDict

VectorD = List[float] ## or any high dimensional number representation, in particualr torch.Tensor

### Optimization Functions
class optimizationFuntion:
    def __init__(self, dim: int, functor: Callable):
        self.dim, self.functor = dim, functor

    def __call__(self, x):
        return self.functor(x)

class optimizationFunctionFactory:
    @staticmethod
    def ratrigin_fun(x: VectorD)->float:
        n = len(x)
        out = 10 * n + sum([(5.12 * x_i) ** 2 - 10 * torch.cos(2 * math.pi * (5.12 * x_i)) for x_i in x])
        return out

    @staticmethod
    def ackley_fun(x: VectorD)->float:
        out = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * sum([(5 * x_i) ** 2 for x_i in x])))
        out -= torch.exp(0.5 * sum([torch.cos(2 * math.pi * 5 * x_i) for x_i in x]))
        out += math.e + 20
        return out

    @classmethod
    def create(cls, type: str, dim : int = 2) -> optimizationFuntion:
        if type == "ackley": return optimizationFuntion( dim, cls.ackley_fun )
        elif type == "ratrigin": return optimizationFuntion( dim, cls.ratrigin_fun )
        raise RuntimeError("Not recognized funciton type")

    @classmethod
    def transform_linearly(cls, alpha: float, beta: float, functor: Callable)->optimizationFuntion:
        def f(x):  return alpha*functor(x) + beta
        return optimizationFuntion(functor.dim, f)


class exploraitonObject:
    class collectionType(Enum):
        NONE = 0,
        ALL = 1,
        IMPROVEMENTS = 2

    def __init__(self, verbose: bool, type_: collectionType = collectionType.IMPROVEMENTS ):
        self.values = defaultdict(OrderedDict)
        self.type = type_
        self.verbose = self.__callVerbose__ if verbose else self.__nothing__

        self.addP1 = self.__add__ if self.type == self.collectionType.ALL else self.__nothing__
        self.addP2 = self.__add__ if self.type == self.collectionType.IMPROVEMENTS else self.__nothing__

    def start(self, x: VectorD, y: float) -> None:
        self.y_best = y
        self.x_best = x

    def __callVerbose__(self, epoch: int, s: int, x: VectorD, y: float)->None:
        print( "epoch ", epoch, "iteration ", s, ", x ", x, ", y ", y)

    def __add__(self, epoch: int, s: int, x: VectorD, y: float)->None:
        self.values[epoch].update({s: ( x, y ) })
        self.verbose(epoch, s, x, y)

    def __nothing__(self, epoch: int, s: int, x: VectorD, y: VectorD)->None: return

    def __call__(self, epoch: int, s: int, x: VectorD, y: VectorD)->None:
        self.addP1(epoch, s, x, y)
        if( y.item() < self.y_best ):
            self.start(x.tolist(), y.item())
            self.addP2(epoch, s, self.x_best, self.y_best)

class optimizationMethods:
    class env:
        def __init__(self, kwargs):
            for k,v in kwargs.items():
                setattr(self, k, v)

    class updateFunctor:
        def __init__(self, locals_to_store: List[str], objs_attr: List[Tuple[str]], func):
            self.locals = locals_to_store
            self.objs = objs_attr
            self.func   = func

        def set_env(self, locals: dict)->None:
            toCache = {y: locals.get(y, None) for y in self.locals}
            toCache.update( {a[1]: getattr( locals.get(a[0], None), a[1], None ) for a in self.objs } )
            self.env = optimizationMethods.env(toCache)

        def __call__(self, x: VectorD)->float:
            return self.func(x, self.env)

    def __init__(self, function: Callable, dim = 2):
        self.func = function if type(function) == optimizationFuntion else \
            optimizationFunctionFactory.create(function, dim)

    def __generalDescent__(self,
                           epochs: int,
                           steps: int,
                           update_rule: updateFunctor,
                           save_evolution: exploraitonObject,
                           initial_rule_per_eppoch: Callable[[int],VectorD] = None)->exploraitonObject:

        save_evolution.start([], float("inf"))
        if not initial_rule_per_eppoch: initial_rule_per_eppoch = self.__defaultInitial__()
        for epoch in range(epochs):
            x = initial_rule_per_eppoch(self.func.dim)
            for s in range(steps):
                y = self.func(x)
                y.backward()
                save_evolution(epoch, s,x,y)
                update_rule.set_env(locals())
                x = update_rule(x)

        return save_evolution

    def __defaultInitial__(self, x0 = 3.9, y0 = 4.0)->Callable[[int],VectorD]:
        def f(d):
            x = torch.FloatTensor(d).uniform_(x0, y0)
            x.requires_grad = True
            return x
        return f

    def gradient_descent(self,
                         lr: float,
                         epochs: int,
                         steps: int,
                         verbose : bool=True,
                         expType: exploraitonObject.collectionType =exploraitonObject.collectionType.IMPROVEMENTS,
                         initial_rule: Callable[[int], VectorD] = None) -> exploraitonObject:
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
                                   lr: float,
                                   epochs: int,
                                   steps: int,
                                   perturbation_ratio: float,
                                   verbose: bool= True,
                                   expType: exploraitonObject.collectionType = exploraitonObject.collectionType.IMPROVEMENTS,
                                   initial_rule: Callable[[int], VectorD] = None) -> exploraitonObject:
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
                                  lr: float,
                                  epochs: int,
                                  steps: int,
                                  perturbation_ratio: float,
                                  gamma: float,
                                  verbose: bool = True,
                                  expType: exploraitonObject.collectionType = exploraitonObject.collectionType.IMPROVEMENTS,
                                  initial_rule: Callable[[int], VectorD] = None) -> exploraitonObject:
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