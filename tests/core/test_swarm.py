from plangym import AtariEnvironment, ParallelEnvironment
from plangym.minimal import ClassicControl
import pytest

from fragile.core.dt_sampler import GaussianDt
from fragile.core.env import BaseEnvironment, DiscreteEnv
from fragile.core.models import BaseModel, DiscreteUniform, NormalContinuous
from fragile.core.swarm import Swarm
from fragile.core.walkers import BaseWalkers, Walkers
from fragile.optimize.benchmarks import Rastrigin
from fragile.optimize.swarm import FunctionMapper


def create_cartpole_swarm():
    swarm = Swarm(
        model=lambda x: DiscreteUniform(env=x),
        walkers=Walkers,
        env=lambda: DiscreteEnv(ClassicControl()),
        reward_limit=150,
        n_walkers=50,
        max_iters=300,
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
    dt = GaussianDt(min_dt=3, max_dt=100, loc_dt=5, scale_dt=2)
    swarm = Swarm(
        model=lambda x: DiscreteUniform(env=x, critic=dt),
        walkers=Walkers,
        env=lambda: DiscreteEnv(env),
        n_walkers=67,
        max_iters=20,
        prune_tree=True,
        reward_scale=2,
        reward_limit=751,
    )
    return swarm


def create_function_swarm():
    shape = (2,)
    env = Rastrigin(shape=shape)
    swarm = FunctionMapper(
        model=lambda x: NormalContinuous(bounds=env.bounds),
        env=lambda: env,
        n_walkers=5,
        max_iters=5,
        prune_tree=True,
        reward_scale=2,
        minimize=False,
    )
    return swarm


swarm_dict = {
    "cartpole": create_cartpole_swarm,
    "atari": create_atari_swarm,
    "function": create_function_swarm,
}


@pytest.fixture()
def swarm(request):
    return TestSwarm.swarm_dict.get(request.param, create_cartpole_swarm)()


class TestSwarm:
    swarm_dict = {
        "cartpole": create_cartpole_swarm,
        "atari": create_atari_swarm,
        "function": create_function_swarm,
    }
    swarm_names = list(swarm_dict.keys())
    test_scores = list(zip(swarm_names, [149, 750, 10]))

    @pytest.mark.parametrize("swarm", swarm_names, indirect=True)
    def test_init_not_crashes(self, swarm):
        assert swarm is not None

    @pytest.mark.parametrize("swarm", swarm_names, indirect=True)
    def test_env_init(self, swarm):
        assert hasattr(swarm.walkers.states, "will_clone")

    @pytest.mark.parametrize("swarm", swarm_names, indirect=True)
    def test_attributes(self, swarm):
        assert isinstance(swarm.env, BaseEnvironment)
        assert isinstance(swarm.model, BaseModel)
        assert isinstance(swarm.walkers, BaseWalkers)

    @pytest.mark.parametrize("swarm", swarm_names, indirect=True)
    def test_reset_no_params(self, swarm):
        swarm.reset()

    @pytest.mark.parametrize("swarm", swarm_names, indirect=True)
    def test_step_does_not_crashes(self, swarm):
        swarm.reset()
        swarm.step_walkers()

    @pytest.mark.parametrize("swarm", swarm_names, indirect=True)
    def test_run(self, swarm):
        swarm.reset()
        swarm.walkers.max_iters = 5
        swarm.run()

    @pytest.mark.parametrize("swarm, target", test_scores, indirect=["swarm"])
    def test_score_gets_higher(self, swarm, target):
        swarm.walkers.seed()
        swarm.reset()
        swarm.walkers.max_iters = 500
        swarm.run()
        reward = swarm.walkers.states.cum_rewards.max()
        assert reward > target, "Iters: {}, rewards: {}".format(
            swarm.walkers.n_iters, swarm.walkers.states.cum_rewards
        )
