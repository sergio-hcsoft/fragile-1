from fragile.core.base_classes import BaseEnvironment
from fragile.core.models import RandomDiscrete
from fragile.core.walkers import Walkers
from fragile.tests.core.fixtures import (
    create_env,
    environment_fact,  # noqa: F401
    plangym_env,
    swarm,
)  # noqa: F401


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
        swarm.reset()

    def test_step(self, swarm):
        swarm.step_walkers()

    def test_run_swarm(self, swarm):
        swarm.run_swarm()

    def test_score_gets_higher(self, swarm):
        swarm.walkers.reset()
        swarm.reset()
        swarm.walkers.max_iters = 500
        swarm.run_swarm()
        reward = swarm.walkers.states.cum_rewards.max()
        assert reward > 100, "Iters: {}, rewards: {}".format(
            swarm.walkers.n_iters, swarm.walkers.states.cum_rewards
        )
