import networkx
from plangym.minimal import ClassicControl
import pytest

from fragile.core import DiscreteEnv, DiscreteUniform, Swarm
from fragile.core.tree import _BaseNetworkxTree, HistoryTree


@pytest.fixture()
def networkx_tree(request):
    return request.param()


@pytest.fixture()
def swarm_with_tree():
    swarm = Swarm(
        model=lambda x: DiscreteUniform(env=x),
        env=lambda: DiscreteEnv(ClassicControl()),
        reward_limit=200,
        n_walkers=150,
        max_iters=300,
        reward_scale=2,
        tree=HistoryTree,
        prune_tree=True,
    )
    return swarm


class TestBaseNetworkxTree:

    networkx_trees = [_BaseNetworkxTree, HistoryTree]

    @pytest.mark.parametrize("networkx_tree", networkx_trees, indirect=True)
    def test_init(self, networkx_tree):
        tree = _BaseNetworkxTree()
        assert isinstance(tree.data, networkx.DiGraph)
        assert tree.ROOT_ID in tree.data.nodes
        assert tree.ROOT_ID in tree.leafs

    @staticmethod
    def test_tree_with_integration_test(swarm_with_tree):
        swarm_with_tree.run()
        assert networkx.is_tree(swarm_with_tree.tree.data)

        best_ix = swarm_with_tree.walkers.states.cum_rewards.argmax()
        best_id = swarm_with_tree.walkers.states.id_walkers[best_ix]
        path = swarm_with_tree.tree.get_branch(best_id, from_hash=True)
