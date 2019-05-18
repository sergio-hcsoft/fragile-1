import pytest
import torch

from fragile.optimize.benchmarks import EggHolder, Rastrigin, Sphere


@pytest.fixture()
def benchmarks():
    data = {
        "sphere": {"env": Sphere, "point": torch.tensor([0.0, 0.0]).view(1, -1), "max_val": 0.0},
        "rastrigin": {
            "env": Rastrigin,
            "point": torch.tensor([0.0, 0.0]).view(1, -1),
            "max_val": 0.0,
        },
        "eggholder": {
            "env": EggHolder,
            "point": torch.tensor([512.0, 404.2319]).view(1, -1),
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
            result = env.function(point)
            assert result.item() == max_val
