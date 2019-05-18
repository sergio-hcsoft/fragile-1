import math

import numpy as np
import torch

from fragile.core.utils import to_numpy, to_tensor
from fragile.optimize.env import Function


class OptimBenchmark(Function):

    benchmark = None
    best_state = None
    function = None

    def __init__(self, shape, **kwargs):
        kwargs = self.process_default_kwargs(shape, kwargs)
        super(OptimBenchmark, self).__init__(**kwargs)

    @staticmethod
    def get_bounds(shape):
        raise NotImplementedError

    @classmethod
    def process_default_kwargs(cls, shape, kwargs):
        kwargs["function"] = kwargs.get("function", cls.function)
        kwargs["shape"] = kwargs.get("shape", shape)
        if kwargs.get("bounds") is None:
            kwargs["bounds"] = cls.get_bounds(shape)
        return kwargs


def sphere(x: torch.Tensor):
    with torch.no_grad():
        return -torch.sum(x ** 2, 1)


class Sphere(OptimBenchmark):
    function = sphere

    @staticmethod
    def get_bounds(shape):
        bounds = [(-1000, 1000) for _ in range(shape[0])]
        return bounds


def rastrigin(x: torch.Tensor):
    with torch.no_grad():
        dims = x.shape[1]
        A = 10
        result = A * dims + torch.sum(x ** 2 - A * torch.cos(2 * math.pi * x), 1)
        return -1 * result


class Rastrigin(OptimBenchmark):
    function = rastrigin

    @staticmethod
    def get_bounds(shape):
        bounds = [(-5.12, 5.12) for _ in range(shape[0])]
        return bounds


def eggholder(tensor: torch.Tensor):
    with torch.no_grad():
        x, y = tensor[:, 0], tensor[:, 1]
        first_root = torch.sqrt(torch.abs(x / 2.0 + (y + 47)))
        second_root = torch.sqrt(torch.abs(x - (y + 47)))
        result = -1 * (y + 47) * torch.sin(first_root) - x * torch.sin(second_root)
        return -1 * result


class EggHolder(OptimBenchmark):
    function = eggholder

    def __init__(self, shape=(2,), **kwargs):
        kwargs = self.process_default_kwargs(shape, kwargs)
        super(OptimBenchmark, self).__init__(**kwargs)

    @staticmethod
    def get_bounds(shape):
        bounds = [(-512, 512), (-512, 512)]
        return bounds

    @classmethod
    def process_default_kwargs(cls, shape, kwargs):
        return super(EggHolder, cls).process_default_kwargs(shape=tuple([2]), kwargs=kwargs)


def styblinski_tang(x):
    with torch.no_grad():
        return -1 * torch.sum(x ** 4 - 16 * x ** 2 + 5 * x, 1) / 2.0


class StyblinskiTang(OptimBenchmark):
    function = styblinski_tang

    @staticmethod
    def get_bounds(shape):
        bounds = [(-5.0, 5.0) for _ in range(shape[0])]
        return bounds


def lj_func(x, n_atoms):
    x = to_numpy(x)

    def lennard_jones(U):
        U = U.reshape(n_atoms, 3)
        npart = len(U)
        Epot = 0.0
        for i in range(npart):
            for j in range(npart):
                if i > j:
                    r2 = np.linalg.norm(U[j, :] - U[i, :]) ** 2
                    r2i = 1.0 / r2
                    r6i = r2i * r2i * r2i
                    Epot = Epot + r6i * (r6i - 1.0)
        Epot = Epot * 4
        return Epot

    result = np.array([lennard_jones(x[i, :]) for i in range(x.shape[0])]).reshape(x.shape[0], 1)
    return -1.0 * to_tensor(result)


class LennardJones(OptimBenchmark):
    minima = {
        "2": -1,
        "3": -3,
        "4": -6,
        "5": -9.103852,
        "6": -12.712062,
        "7": -16.505384,
        "8": -19.821489,
        "9": -24.113360,
        "10": -28.422532,
        "11": -32.765970,
        "12": -37.967600,
        "13": -44.326801,
        "14": -47.845157,
        "15": -52.322627,
    }

    benchmark = None

    def __init__(self, n_atoms: int = 10, *args, **kwargs):
        self.n_atoms = n_atoms
        shape = (3 * n_atoms,)
        print(shape)
        self.benchmark = [np.zeros(self.n_atoms * 3), self.minima[str(n_atoms)]]
        super(LennardJones, self).__init__(shape=shape, *args, **kwargs)

        def lennard_jones(x):
            return lj_func(x, self.n_atoms)

        self.function = lennard_jones

    @staticmethod
    def get_bounds(shape):
        bounds = [(-1.1, 1.1) for _ in range(shape[0])]
        return bounds
