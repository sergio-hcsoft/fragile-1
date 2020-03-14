from typing import Callable, Tuple

import numpy
from plangym import AtariEnvironment
import pytest

from fragile.atari import AtariEnv
from fragile.core.states import StatesEnv, StatesModel
from tests.core.test_env import TestEnvironment


N_WALKERS = 10


def pacman_ram():
    env = AtariEnvironment(name="MsPacman-ram-v0", clone_seeds=True, autoreset=True)
    env.reset()
    env = AtariEnv(env)
    return env


def qbert_rgb():
    env = AtariEnvironment(name="Qbert-v0", clone_seeds=True, autoreset=True)
    env.reset()
    env = AtariEnv(env)
    return env


def create_env_and_model_states(name="classic") -> Callable:
    def _pacman_ram():
        env = pacman_ram()
        params = {"actions": {"dtype": numpy.int64}, "critic": {"dtype": numpy.float32}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        states.update(actions=numpy.ones(N_WALKERS), critic=numpy.ones(N_WALKERS))
        return env, states

    def _qbert_rgb():
        env = qbert_rgb()
        params = {"actions": {"dtype": numpy.int64}, "critic": {"dtype": numpy.float32}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        states.update(actions=numpy.ones(N_WALKERS), critic=numpy.ones(N_WALKERS))
        return env, states

    if name.lower() == "pacman_ram":
        return _pacman_ram
    elif name.lower() == "qbert_rgb":
        return _qbert_rgb


env_fixture_params = ["qbert_rgb", "pacman_ram"]


class TestAtari(TestEnvironment):
    @pytest.fixture(params=env_fixture_params)
    def env_data(self, request) -> Tuple[AtariEnv, StatesModel]:
        if request.param in env_fixture_params:
            env, model_states = create_env_and_model_states(request.param)()

        else:
            raise ValueError("Environment not well defined: %s" % request.param)
        return env, model_states
