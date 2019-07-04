import numpy as np
import pytest

from fragile.optimize.mapper import FunctionMapper
from fragile.optimize.models import RandomNormal


@pytest.fixture()
def swarm():
    def potential_well(x):
        return -np.sum((x - 1) ** 2, 1) - 1

    bounds = [(-10, 10), (-5, 5)]

    def model(x):
        return RandomNormal(
            high=np.array([100, 100]), low=np.array([-100, -100]), env=x, shape=None
        )

    return FunctionMapper.from_function(
        n_vectors=5,
        function=potential_well,
        bounds=bounds,
        model=model,
        shape=(2,),
        n_walkers=10,
        reward_scale=1,
        accumulate_rewards=False,
    )


@pytest.fixture()
def finished_swarm(swarm):
    swarm.walkers.reset()
    swarm.init_walkers()
    swarm.walkers.max_iters = 500
    swarm.run_swarm()
    return swarm


class TestFunctionMapper:
    def test_init(self, swarm):
        pass

    def test_init_walkers_no_params(self, swarm):
        swarm.init_walkers()

    def test_step(self, swarm):
        swarm.step_walkers()

    def test_run_swarm(self, swarm):
        swarm.run_swarm()

    def test_score_gets_higher(self, finished_swarm):
        reward = finished_swarm.walkers.cum_rewards.max().item()
        assert reward >= -60, "Iters: {}, rewards: {}".format(
            finished_swarm.walkers.n_iters, finished_swarm.walkers.cum_rewards
        )

    def test_has_vector(self, finished_swarm):
        pass
        # assert isinstance(finished_swarm.walkers.encoder.vectors[0], Vector)
