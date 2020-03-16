import sys
from typing import Callable, Tuple

import numpy as numpy
import pytest

from fragile.core.env import Environment
from fragile.core.states import StatesModel
from fragile.distributed.env import ParallelEnvironment, ParallelFunction, RayEnv, RayFunction
from fragile.optimize.benchmarks import Rastrigin
from tests.core.test_env import discrete_atari_env, classic_control_env, TestEnvironment
from tests.distributed.ray import init_ray, ray
from tests.optimize.test_env import Function, TestFunction, local_minimizer

N_WALKERS = 10


def ray_env():
    env = RayEnv(classic_control_env, n_workers=1)
    return env


def ray_function():
    return RayFunction(local_minimizer, n_workers=2)


def parallel_environment():
    return ParallelEnvironment(discrete_atari_env, n_workers=2)


def parallel_function():
    return ParallelFunction(env_callable=lambda: Rastrigin(dims=2), n_workers=2)


def create_env_and_model_states(name="classic") -> Callable:
    def _ray_env():
        init_ray()
        env = ray_env()
        params = {"actions": {"dtype": numpy.int64}, "critic": {"dtype": numpy.float32}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        return env, states

    def _ray_function():
        init_ray()
        env = ray_function()
        params = {"actions": {"dtype": numpy.int64}, "critic": {"dtype": numpy.float32}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        return env, states

    def _parallel_function():
        env = parallel_function()
        params = {
            "actions": {"dtype": numpy.float32, "size": (2,)},
            "critic": {"dtype": numpy.float32},
        }
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        return env, states

    def _parallel_environment():
        env = parallel_environment()
        params = {"actions": {"dtype": numpy.int64}, "critic": {"dtype": numpy.float32}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        states.update(actions=numpy.ones(N_WALKERS), critic=numpy.ones(N_WALKERS))
        return env, states

    if name.lower() == "ray_env":
        return _ray_env
    elif name.lower() == "ray_function":
        return _ray_function
    elif name.lower() == "parallel_function":
        return _parallel_function
    elif name.lower() == "parallel_environment":
        return _parallel_environment


env_fixture_params = ["parallel_function", "parallel_environment"]


class TestDistributedEnvironment(TestEnvironment):
    @pytest.fixture(params=env_fixture_params, scope="class")
    def env_data(self, request) -> Tuple[Environment, StatesModel]:
        if request.param in env_fixture_params:
            env, model_states = create_env_and_model_states(request.param)()
            if "ray" in request.param:

                def kill_ray_env():
                    try:
                        for e in env.envs:
                            e.__ray_terminate__.remote()
                    except AttributeError:
                        pass
                    ray.shutdown()

                request.addfinalizer(kill_ray_env)

            elif "parallel" in request.param:

                def kill_parallel_env():
                    env.close()

                request.addfinalizer(kill_parallel_env)

        else:
            raise ValueError("Environment not well defined: %s" % request.param)
        return env, model_states


ray_env_fixture_params = ["ray_env", "ray_function"]


@pytest.mark.skipif(sys.version_info >= (3, 8), reason="Requires python3.7 or lower")
class TestDistributedFunction(TestFunction):
    @pytest.fixture(params=ray_env_fixture_params)
    def env_data(self, request):
        if request.param in ray_env_fixture_params:
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
