import numpy as np
import pytest

from fragile.optimize.benchmarks import EggHolder, Rastrigin, Sphere


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


class TestBenchmarks:
    def test_2_dim_optimum(self, benchmarks):
        shape = tuple([2])
        for bench in benchmarks.values():
            env_cls, point, max_val = list(bench.values())
            env = env_cls(shape=shape)
            result = env.function(point)[0]
            print(result, env_cls.__name__)
            assert np.allclose(result, max_val)
