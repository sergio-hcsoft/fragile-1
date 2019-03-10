import pytest
import numpy as np
from plangym import AtariEnvironment, ClassicControl
from fragile.env import DiscreteEnv
from fragile.states import BaseStates


@pytest.fixture(scope="module")
def create_env(name="classic"):
    def atari_env():
        env = AtariEnvironment(name="MsPacman-v0", clone_seeds=True, autoreset=True)

        _ = env.reset()
        return env

    def classic_control_env():
        env = ClassicControl()
        _ = env.reset()
        return env

    if name.lower() == "atari":
        return atari_env
    else:
        return classic_control_env


@pytest.fixture(scope="module")
def plangym_env(create_env):
    env = create_env()
    _ = env.reset()
    return env


@pytest.fixture(scope="module")
def environment(plangym_env):
    env = DiscreteEnv(plangym_env)
    return env


class TestEnvironment:
    def test_reset(self, environment):
        states = environment.reset()
        assert isinstance(states, BaseStates), states

        batch_size = 10
        states = environment.reset(batch_size=batch_size)
        assert isinstance(states, BaseStates), states
        assert states.observs.shape[0] == batch_size
        assert states.rewards.shape[0] == batch_size
        assert states.ends.shape[1] == 1

    def test_step(self, environment):
        batch_size = 17
        actions = np.zeros(batch_size, dtype=int)
        states = environment.reset(batch_size=batch_size)
        new_states = environment.step(actions=actions, env_states=states)
        assert new_states.ends.sum().cpu().item() == 0
