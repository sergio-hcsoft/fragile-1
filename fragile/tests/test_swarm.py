import pytest

from fragile.core.base_classes import BaseEnvironment
from fragile.core.env import DiscreteEnv
from fragile.core.models import RandomDiscrete
from fragile.core.swarm import Swarm
from fragile.core.walkers import Walkers
from fragile.tests.test_env import create_env, plangym_env  # noqa: F401


@pytest.fixture(scope="module")
def environment_fact(plangym_env):
    env = DiscreteEnv(plangym_env)
    return lambda: env


@pytest.fixture(scope="module")
def swarm(environment_fact):
    n_walkers = 50
    swarm = Swarm(
        model=lambda x: RandomDiscrete(x),
        env=environment_fact,
        walkers=Walkers,
        n_walkers=n_walkers,
        max_iters=10,
        use_tree=True,
    )
    return swarm


class TestSwarm:
    def test_init_not_crashes(self, swarm):
        assert swarm is not None

    def test_env_init(self, swarm):
        assert hasattr(swarm.walkers, "observs")
        assert hasattr(swarm.walkers, "rewards")
        assert hasattr(swarm.walkers, "ends")
        assert hasattr(swarm.walkers, "will_clone")

    def test_attributes(self, swarm):
        assert isinstance(swarm.env, BaseEnvironment)
        assert isinstance(swarm.model, RandomDiscrete)
        assert isinstance(swarm.walkers, Walkers)

    def test_init_walkers_no_params(self, swarm):
        swarm.init_walkers()

    def test_step(self, swarm):
        swarm.step_walkers()

    def test_run_swarm(self, swarm):
        swarm.run_swarm()

    def test_score_gets_higher(self, swarm):
        swarm.walkers.reset()
        swarm.init_walkers()
        swarm.walkers.max_iters = 500
        swarm.run_swarm()
        reward = swarm.walkers.cum_rewards.max()
        assert reward > 100, "Iters: {}, rewards: {}".format(
            swarm.walkers.n_iters, swarm.walkers.cum_rewards
        )
