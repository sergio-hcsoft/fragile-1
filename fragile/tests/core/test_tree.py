import pytest  # noqa: F401

from fragile.core.base_classes import States
from fragile.core.tree import Node, Tree
from fragile.tests.core.fixtures import (
    create_env,
    env_and_model_states,  # noqa: F401
    environment_fact,
    finished_tree,
    node,
    node_id,
    plangym_env,
    swarm,
    tree,  # noqa: F401
    walkers_factory,
)  # noqa: F401


class TestNode:
    def test_init(self, node):
        assert isinstance(node.env_state, States)
        assert isinstance(node.model_state, States)


class TestTree:
    def test_init(self, tree):
        assert isinstance(tree(), Tree)

    def test_reset(self, tree, env_and_model_states):
        tree = tree()
        env_state, model_state = env_and_model_states
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
        for nod_id in path:
            assert isinstance(t.nodes[nod_id].env_state, States)

    def test_prune_branch(self, finished_tree):
        t, swarm = finished_tree
        max_reward = int(swarm.walkers.states.cum_rewards.argmax())
        best_id = swarm.walkers.states.id_walkers[max_reward]
        t.prune_branch(best_id)
        assert best_id not in t.nodes.keys()
