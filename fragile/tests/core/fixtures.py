from typing import Callable

import numpy as np
from plangym import AtariEnvironment, ClassicControl
from plangym.env import Environment
import pytest

from fragile.core.env import DiscreteEnv
from fragile.core.models import RandomContinous, RandomDiscrete
from fragile.core.states import States
from fragile.core.swarm import Swarm
from fragile.core.tree import Node, Tree
from fragile.core.walkers import StatesWalkers, Walkers


@pytest.fixture()
def states_from_tensor() -> States:
    n_walkers = 10
    miau = np.ones((10, 1, 100))
    miau_2 = np.zeros((10, 33, 1))
    return States(batch_size=n_walkers, miau=miau, miau_2=miau_2)


@pytest.fixture()
def discrete_model() -> RandomDiscrete:
    return RandomDiscrete(n_actions=10)


@pytest.fixture()
def continous_model() -> RandomContinous:
    return RandomContinous(low=-1, high=1, shape=(3,))


@pytest.fixture(scope="module")
def environment_fact(plangym_env) -> Callable:
    env = DiscreteEnv(plangym_env)
    return lambda: env


@pytest.fixture(scope="module")
def walkers():
    n_walkers = 10
    env_dict = {
        "env_1": {"size": (1, 100)},
        "env_2": {"size": (1, 33)},
        "observs": {"size": (1, 100)},
    }
    model_dict = {"model_1": {"size": (1, 13)}, "model_2": {"size": (1, 5)}}

    walkers = Walkers(
        n_walkers=n_walkers, env_state_params=env_dict, model_state_params=model_dict
    )
    return walkers


@pytest.fixture(scope="module")
def walkers_factory():
    def new_walkers():
        n_walkers = 10
        env_dict = {
            "env_1": {"size": (1, 100)},
            "env_2": {"size": (1, 33)},
            "observs": {"size": (1, 100)},
        }
        model_dict = {"model_1": {"size": (1, 13)}, "model_2": {"size": (1, 5)}}

        walkers = Walkers(
            n_walkers=n_walkers, env_state_params=env_dict, model_state_params=model_dict
        )
        return walkers

    return new_walkers


@pytest.fixture(scope="module")
def states_walkers():
    return StatesWalkers(10)



@pytest.fixture()
def node_id():
    return int(np.random.randint(0, 1000))


@pytest.fixture()
def node(env_and_model_states, node_id):
    env_state, model_state = env_and_model_states
    return Node(
        node_id=node_id,
        parent_id=node_id - 1,
        env_state=env_state,
        model_state=model_state,
        reward=node_id + 10,
    )


@pytest.fixture()
def tree():
    return lambda: Tree()


@pytest.fixture()
def finished_tree(swarm):
    swarm.run_swarm()
    return swarm.tree, swarm


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
