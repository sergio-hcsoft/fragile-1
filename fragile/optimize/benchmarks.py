import math

import torch

from fragile.optimize.env import Function


class OptimBenchmark(Function):
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
