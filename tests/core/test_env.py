from typing import Callable, Tuple

import numpy as np
from plangym import AtariEnvironment, ClassicControl
import pytest

from fragile.core.bounds import Bounds
from fragile.core.env import DiscreteEnv, Environment
from fragile.core.states import States
from fragile.optimize.env import Function

N_WALKERS = 10


def create_env_and_model_states(name="classic") -> Callable:
    def atari_env():
        env = AtariEnvironment(name="MsPacman-v0", clone_seeds=True, autoreset=True)
        env.reset()
        env = DiscreteEnv(env)
        params = {"actions": {"dtype": np.int64}, "critic": {"dtype": np.float32}}
        states = States(state_dict=params, batch_size=N_WALKERS)
        states.update(actions=np.ones(N_WALKERS), critic=np.ones(N_WALKERS))
        return env, states

    def classic_control_env():
        env = ClassicControl()
        env.reset()
        env = DiscreteEnv(env)
        params = {"actions": {"dtype": np.int64}, "dt": {"dtype": np.float32}}
        states = States(state_dict=params, batch_size=N_WALKERS)
        states.update(actions=np.ones(N_WALKERS), dt=np.ones(N_WALKERS))
        return env, states

    def function_env():
        bounds = Bounds(shape=(2,), high=1, low=1, dtype=int)
        env = Function(function=lambda x: np.ones(N_WALKERS), bounds=bounds)
        params = {"actions": {"dtype": np.int64, "size": (2,)}, "dt": {"dtype": np.float32}}
        states = States(state_dict=params, batch_size=N_WALKERS)
        return env, states

    if name.lower() == "pacman":
        return atari_env
    elif name.lower() == "function":
        return function_env
    else:
        return classic_control_env


@pytest.fixture(scope="module")
def env_data(request) -> Tuple[Environment, States]:
    if request.param in TestBaseEnvironment.env_fixtures_params:
        env, model_states = create_env_and_model_states(request.param)()

    else:
        raise ValueError("Environment not well defined")
    return env, model_states


class TestBaseEnvironment:
    env_fixtures_params = ["classic", "pacman", "function"]

    @pytest.mark.parametrize("env_data", env_fixtures_params, indirect=True)
    def test_reset(self, env_data):
        env, model_states = env_data
        states = env.reset()
        assert isinstance(states, env.STATE_CLASS), states

        batch_size = 10
        states = env.reset(batch_size=batch_size)
        assert isinstance(states, env.STATE_CLASS), states
        for name in states.keys():
            assert states[name].shape[0] == batch_size

    @pytest.mark.parametrize("env_data", env_fixtures_params, indirect=True)
    def test_get_params_dir(self, env_data):
        env, model_states = env_data
        params_dict = env.get_params_dict()
        assert isinstance(params_dict, dict)
        for k, v in params_dict.items():
            assert isinstance(k, str)
            assert isinstance(v, dict)
            for ki in v.keys():
                assert isinstance(ki, str)

    @pytest.mark.parametrize("env_data", env_fixtures_params, indirect=True)
    def test_step(self, env_data):
        batch_size = 10
        env, model_states = env_data
        states = env.reset(batch_size=batch_size)
        new_states = env.step(model_states=model_states, env_states=states)
        assert new_states.ends.sum() == 0
