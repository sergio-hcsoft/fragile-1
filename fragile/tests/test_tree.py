import numpy as np
import pytest

from fragile.core.base_classes import BaseStates
from fragile.core.tree import Node, Tree
from fragile.tests.test_swarm import create_env, environment_fact, plangym_env, swarm  # noqa: F401
from fragile.tests.test_walkers import walkers_factory  # noqa: F401


@pytest.fixture(scope="module")
def states(walkers_factory):
    env_state = list(walkers_factory().env_states.split_states())[0]
    model_state = list(walkers_factory().model_states.split_states())[0]
    return env_state, model_state


@pytest.fixture()
def node_id():
    return int(np.random.randint(0, 1000))


@pytest.fixture()
def node(states, node_id):
    env_state, model_state = states
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


class TestNode:
    def test_init(self, node):
        assert isinstance(node.env_state, BaseStates)
        assert isinstance(node.model_state, BaseStates)


class TestTree:
    def test_init(self, tree):
        assert isinstance(tree(), Tree)

    def test_reset(self, tree, states):
        tree = tree()
        env_state, model_state = states
        tree.reset(env_state=env_state, model_state=model_state, reward=0)
        assert tree.curr_id == 0
        assert isinstance(tree.nodes[tree.curr_id], Node)
        assert tree.nodes[tree.curr_id].env_state is not None
        assert tree.nodes[tree.curr_id].model_state is not None

    def test_path(self, finished_tree):

        t, swarm = finished_tree
        max_reward = int(swarm.walkers.states.cum_rewards.argmax())
        best_id = swarm.walkers.states.id_walkers[max_reward]
        path = t.get_path_ids(best_id)

        assert len(path) > 1
        assert isinstance(path, list)
        assert t.nodes[path[-1]].reward >= t.nodes[path[0]].reward
        for node_id in path:
            assert isinstance(t.nodes[node_id].env_state, BaseStates)

    def test_prune_branch(self, finished_tree):
        t, swarm = finished_tree
        max_reward = int(swarm.walkers.states.cum_rewards.argmax())
        best_id = swarm.walkers.states.id_walkers[max_reward]
        t.prune_branch(best_id)
        assert best_id not in t.nodes.keys()
