from typing import Callable

import numpy as np
import pytest

from plangym import AtariEnvironment, ClassicControl

from fragile.core.env import Environment, DiscreteEnv
from fragile.core.states import States


def create_env(name="classic") -> Callable:
    def atari_env():
        env = AtariEnvironment(name="MsPacman-v0", clone_seeds=True, autoreset=True)
        env.reset()
        return env

    def classic_control_env():
        env = ClassicControl()
        env.reset()
        return env

    if name.lower() == "pacman":
        return atari_env
    else:
        return classic_control_env


@pytest.fixture(scope="module")
def env(request) -> Environment:
    if request.param in ["classic", "pacman"]:
        env = DiscreteEnv(create_env(request.param)())
    return env


env_fixtures_params = ["classic", "pacman"]


class TestBaseEnvironment:
    @pytest.mark.parametrize("env", ["classic", "pacman"], indirect=True)
    def test_reset(self, env):
        states = env.reset()
        assert isinstance(states, States), states

        batch_size = 10
        states = env.reset(batch_size=batch_size)
        assert isinstance(states, env.STATE_CLASS), states
        for name in states.keys():
            assert states[name].shape[0] == batch_size

    @pytest.mark.parametrize("env", ["classic", "pacman"], indirect=True)
    def test_get_params_dir(self, env):
        params_dict = env.get_params_dict()
        assert isinstance(params_dict, dict)
        for k, v in params_dict.items():
            assert isinstance(k, str)
            assert isinstance(v, dict)
            for ki, vi in v.items():
                assert isinstance(ki, str)

    """
    def test_step(self, environment):
        batch_size = 17
        model_state = States(
            17, actions=np.zeros(batch_size, dtype=int), dt=np.ones(batch_size)
        )
        states = environment.reset(batch_size=batch_size)
        new_states = environment.step(model_states=model_state, env_states=states)
        assert new_states.ends.sum() == 0
    """
