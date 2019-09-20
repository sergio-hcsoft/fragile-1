import itertools

import numpy as np
import pytest

from fragile.optimize.benchmarks import (
    EggHolder,
    OptimBenchmark,
    Rastrigin,
    Sphere,
    StyblinskiTang,
)


@pytest.fixture()
def benchmarks():
    data = {
        "sphere": {"env": Sphere, "point": np.zeros((1, 2)), "max_val": 0.0},
        "rastrigin": {"env": Rastrigin, "point": np.zeros((1, 2)), "max_val": 0.0},
        "eggholder": {
            "env": EggHolder,
            "point": np.array([512.0, 404.2319]).reshape(1, -1),
            "max_val": 959.6406860351562,
        },
    }
    return data


@pytest.fixture()
def wiki_benchmark(request) -> OptimBenchmark:
    print(request.param)
    cls, shape = request.param
    return cls(shape=shape)


class TestBenchmarks:
    wiki_bench_classes = [EggHolder, Rastrigin, Sphere, StyblinskiTang]

    @pytest.mark.parametrize(
        "wiki_benchmark", list(itertools.product(wiki_bench_classes, [(2,)])), indirect=True
    )
    def test_optimim(self, wiki_benchmark):
        best = wiki_benchmark.best_state
        new_shape = (1,) + tuple(best.shape)
        val = wiki_benchmark.function(best.reshape(new_shape))
        bench = wiki_benchmark.benchmark
        assert np.allclose(val[0], bench), wiki_benchmark.__class__.__name__
