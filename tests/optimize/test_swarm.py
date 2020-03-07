"""
import numpy as numpy
import pytest

from fragile.optimize.mapper import FunctionMapper
from fragile.optimize.models import NormalContinuous


@pytest.fixture()
def mapper():
    def potential_well(x):
        return -numpy.sum((x - 1) ** 2, 1) - 1

    bounds = [(-10, 10), (-5, 5)]

    def model(x):
        return NormalContinuous(
            high=numpy.array([100, 100]), low=numpy.array([-100, -100]), env=x, shape=None
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
def finished_swarm(mapper):
    mapper.walkers.reset()
    mapper.reset()
    mapper.walkers.max_iters = 500
    mapper.run_swarm()
    return mapper


class TestFunctionMapper:
    def test_init(self, mapper):
        pass

    def test_init_walkers_no_params(self, mapper):
        mapper.reset()

    def test_step(self, mapper):
        mapper.step_walkers()

    def test_run_swarm(self, mapper):
        mapper.run_swarm()

    def test_score_gets_higher(self, finished_swarm):
        reward = finished_swarm.walkers.cum_rewards.max().item()
        assert reward >= -60, "Iters: {}, rewards: {}".format(
            finished_swarm.walkers.n_iters, finished_swarm.walkers.cum_rewards
        )

    def test_has_vector(self, finished_swarm):
        pass
        # assert isinstance(finished_swarm.walkers.critic.vectors[0], Vector)

"""
