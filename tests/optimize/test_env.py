from typing import Callable
import numpy
import pytest

from fragile.core import Bounds
from fragile.core.states import StatesEnv, StatesModel
from fragile.optimize.benchmarks import sphere
from fragile.optimize.env import Function, MinimizerWrapper
from tests.core.test_env import TestEnvironment

N_WALKERS = 50


def function() -> Function:
    return Function.from_bounds_params(
        function=sphere, shape=(2,), low=numpy.array([-10, -5]), high=numpy.array([10, 5]),
    )


def local_minimizer():
    bounds = Bounds(shape=(2,), high=10, low=-5, dtype=float)
    env = Function(function=sphere, bounds=bounds)
    return MinimizerWrapper(env)


def custom_domain_function():
    bounds = Bounds(shape=(2,), high=10, low=-5, dtype=float)
    env = Function(
        function=sphere, bounds=bounds, custom_domain_check=lambda x: numpy.linalg.norm(x) < 5.0
    )
    return env


def create_env_and_model_states(name="classic") -> Callable:
    def _function():
        env = function()
        params = {"actions": {"dtype": numpy.float64, "size": (2,)}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        return env, states

    def _local_minimizer():
        env = local_minimizer()
        params = {"actions": {"dtype": numpy.float64, "size": (2,)}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        return env, states

    def _custom_domain_function():
        env = custom_domain_function()
        params = {"actions": {"dtype": numpy.float64, "size": (2,)}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        return env, states

    if name.lower() == "function":
        return _function
    elif name.lower() == "custom_domain_function":
        return _custom_domain_function
    elif name.lower() == "local_minimizer":
        return _local_minimizer


env_fixture_params = ["local_minimizer", "function", "custom_domain_function"]


class TestFunction(TestEnvironment):
    @pytest.fixture(params=env_fixture_params)
    def env_data(self, request):
        if request.param in env_fixture_params:
            env, model_states = create_env_and_model_states(request.param)()

        else:
            raise ValueError("Environment not well defined: %s" % request.param)
        return env, model_states

    @pytest.fixture()
    def dummy_env(self) -> Function:
        return Function.from_bounds_params(
            function=lambda x: numpy.ones(len(x)),
            shape=(2,),
            low=numpy.array([-10, -5]),
            high=numpy.array([10, 5]),
        )

    def test_init_error(self):
        with pytest.raises(TypeError):
            Function(function=sphere, bounds=(True, False))

    def test_from_bounds_params_error(self):
        with pytest.raises(TypeError):
            Function.from_bounds_params(function=sphere)

    @pytest.mark.parametrize("batch_size", [1, 10])
    def test_reset_batch_size(self, dummy_env, batch_size):
        new_states: StatesEnv = dummy_env.reset(batch_size=batch_size)
        assert isinstance(new_states, StatesEnv)
        assert not (new_states.observs == 0).all().item()
        assert (new_states.rewards == 1).all().item(), (
            new_states.rewards,
            new_states.rewards.shape,
        )
        assert (new_states.oobs == 0).all().item()
        assert len(new_states.rewards.shape) == 1
        assert new_states.rewards.shape[0] == batch_size
        assert new_states.oobs.shape[0] == batch_size
        assert new_states.observs.shape[0] == batch_size
        assert new_states.observs.shape[1] == 2

    def test_step(self, dummy_env):
        states = dummy_env.reset()
        actions = StatesModel(actions=numpy.ones((1, 2)) * 2, batch_size=1, dt=numpy.ones((1, 2)))
        new_states: StatesEnv = dummy_env.step(actions, states)
        assert isinstance(new_states, StatesEnv)
        assert new_states.rewards[0].item() == 1

    def test_minimizer_getattr(self):
        bounds = Bounds(shape=(2,), high=10, low=-5, dtype=float)
        env = Function(function=sphere, bounds=bounds)
        minim = MinimizerWrapper(env)
        assert minim.shape == env.shape

    def test_minimizer_step(self):
        minim = local_minimizer()
        params = {"actions": {"dtype": numpy.float64, "size": (2,)}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        assert minim.shape == minim.shape
        states = minim.step(model_states=states, env_states=minim.reset(N_WALKERS))
        assert numpy.allclose(states.rewards.min(), 0)
