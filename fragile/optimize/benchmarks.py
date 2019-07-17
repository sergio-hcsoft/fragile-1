import math

from numba import jit
import numpy as np

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


def sphere(x: np.ndarray):
    return -np.sum(x ** 2, 1).flatten()


class Sphere(OptimBenchmark):
    function = sphere

    @staticmethod
    def get_bounds(shape):
        bounds = [(-1000, 1000) for _ in range(shape[0])]
        return bounds


def rastrigin(x: np.ndarray):
    dims = x.shape[1]
    A = 10
    result = A * dims + np.sum(x ** 2 - A * np.cos(2 * math.pi * x), 1)
    return 1 * result.flatten()


class Rastrigin(OptimBenchmark):
    function = rastrigin

    @staticmethod
    def get_bounds(shape):
        bounds = [(-5.12, 5.12) for _ in range(shape[0])]
        return bounds


def eggholder(tensor: np.ndarray):
    x, y = tensor[:, 0], tensor[:, 1]
    first_root = np.sqrt(np.abs(x / 2.0 + (y + 47)))
    second_root = np.sqrt(np.abs(x - (y + 47)))
    result = -1 * (y + 47) * np.sin(first_root) - x * np.sin(second_root)
    return result


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
    return np.sum(x ** 4 - 16 * x ** 2 + 5 * x, 1) / 2.0


class StyblinskiTang(OptimBenchmark):
    function = styblinski_tang

    @staticmethod
    def get_bounds(shape):
        bounds = [(-5.0, 5.0) for _ in range(shape[0])]
        return bounds


@jit(nopython=True)
def lennard_fast(state):
    state = state.reshape(-1, 3)
    npart = len(state)
    epot = 0.0
    for i in range(npart):
        for j in range(npart):
            if i > j:
                r2 = np.sum((state[j, :] - state[i, :]) ** 2)
                r2i = 1.0 / r2
                r6i = r2i * r2i * r2i
                epot = epot + r6i * (r6i - 1.0)
    epot = epot * 4
    return epot


@jit(nopython=True)
def numba_lennard(x):
    result = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        result[i] = lennard_fast(x[i])
    return result


def lennard_jones(x: np.ndarray):
    result = -1 * numba_lennard(x)
    return result


class LennardJones(OptimBenchmark):
    # http://doye.chem.ox.ac.uk/jon/structures/LJ/tables.150.html
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
        "20": -77.177043,
        "25": -102.372663,
        "30": -128.286571,
        "38": -173.928427,
        "50": -244.549926,
        "100": -557.039820,
        "104": -582.038429,
    }

    benchmark = None

    def __init__(self, n_atoms: int = 10, *args, **kwargs):
        self.n_atoms = n_atoms
        shape = (3 * n_atoms,)
        print(shape)
        self.benchmark = [np.zeros(self.n_atoms * 3), self.minima.get(str(int(n_atoms)), 0)]
        super(LennardJones, self).__init__(shape=shape, *args, **kwargs)

        self.function = lennard_jones

    @staticmethod
    def get_bounds(shape):
        bounds = [(-1.5, 1.5) for _ in range(shape[0])]
        return bounds

    def boundary_condition(self, points, rewards):
        ends = super(LennardJones, self).boundary_condition(points, rewards)
        mean = rewards.mean()
        too_bad = rewards < mean  # -2_000_000  #
        if int(too_bad.sum()) < len(too_bad):
            ends[too_bad] = 1
        else:
            print(too_bad, rewards)
        return ends
