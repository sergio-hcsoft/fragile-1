from plangym import AtariEnvironment, ParallelEnvironment
from plangym.minimal import ClassicControl
import pytest

from fragile.core.env import BaseEnvironment, DiscreteEnv
from fragile.core.models import RandomDiscrete
from fragile.core.swarm import Swarm
from fragile.core.walkers import Walkers


def create_cartpole_swarm():
    swarm = Swarm(
        model=lambda x: RandomDiscrete(x),
        walkers=Walkers,
        env=lambda: DiscreteEnv(ClassicControl()),
        n_walkers=15,
        max_iters=200,
        prune_tree=True,
        reward_scale=2,
    )
    return swarm


def create_atari_swarm():
    env = ParallelEnvironment(
        env_class=AtariEnvironment,
        name="MsPacman-ram-v0",
        clone_seeds=True,
        autoreset=True,
        blocking=False,
    )

    swarm = Swarm(
        model=lambda x: RandomDiscrete(x),
        walkers=Walkers,
        env=lambda: DiscreteEnv(env),
        n_walkers=67,
        max_iters=20,
        prune_tree=True,
        reward_scale=2,
    )
    return swarm


swarm_dict = {"cartpole": create_cartpole_swarm, "atari": create_atari_swarm}


@pytest.fixture()
def swarm(request):
    return swarm_dict.get(request.param, create_cartpole_swarm)()


class TestSwarm:

    swarm_names = ["cartpole", "atari"]
    test_scores = list(zip(swarm_names, [149, 750]))

    @pytest.mark.parametrize("swarm", swarm_names, indirect=True)
    def test_init_not_crashes(self, swarm):
        assert swarm is not None

    @pytest.mark.parametrize("swarm", swarm_names, indirect=True)
    def test_env_init(self, swarm):
        assert hasattr(swarm.walkers.states, "will_clone")

    @pytest.mark.parametrize("swarm", swarm_names, indirect=True)
    def test_attributes(self, swarm):
        assert isinstance(swarm.env, BaseEnvironment)
        assert isinstance(swarm.model, RandomDiscrete)
        assert isinstance(swarm.walkers, Walkers)

    @pytest.mark.parametrize("swarm", swarm_names, indirect=True)
    def test_reset_no_params(self, swarm):
        swarm.reset()

    @pytest.mark.parametrize("swarm", swarm_names, indirect=True)
    def test_step_does_not_crashes(self, swarm):
        swarm.reset()
        swarm.step_walkers()

    @pytest.mark.parametrize("swarm", swarm_names, indirect=True)
    def test_run_swarm(self, swarm):
        swarm.reset()
        swarm.walkers.max_iters = 5
        swarm.run_swarm()

    @pytest.mark.parametrize("swarm, target", test_scores, indirect=["swarm"])
    def test_score_gets_higher(self, swarm, target):
        swarm.walkers.seed()
        swarm.reset()
        swarm.walkers.max_iters = 150
        swarm.run_swarm()
        reward = swarm.walkers.states.cum_rewards.max()
        assert reward > target, "Iters: {}, rewards: {}".format(
            swarm.walkers.n_iters, swarm.walkers.states.cum_rewards
        )
