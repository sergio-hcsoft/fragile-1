import networkx
import pytest

from fragile.core.tree import _BaseNetworkxTree, HistoryTree


@pytest.fixture()
def networkx_tree(request):
    return request.param()


class TestBaseNetworkxTree:

    networkx_trees = [_BaseNetworkxTree, HistoryTree]

    @pytest.mark.parametrize("networkx_tree", networkx_trees, indirect=True)
    def test_init(self, networkx_tree):
        tree = _BaseNetworkxTree()
        assert isinstance(tree.data, networkx.DiGraph)
        assert tree.ROOT_ID in tree.data.nodes
        assert tree.ROOT_ID in tree.leafs

    @pytest.mark.parametrize("networkx_tree", networkx_trees, indirect=True)
    def test_get_update_hash(self, networkx_tree):
        pass
