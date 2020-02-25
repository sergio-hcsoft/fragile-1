import copy
from typing import List, Union

import networkx as nx

from fragile.core.base_classes import States, BaseStateTree
from fragile.core.walkers import StatesWalkers


class _BaseNetworkxTree(BaseStateTree):
    """This is a tree data structure that stores the paths followed by the walkers. It can be
    pruned to delete paths that are longer be needed. It uses a networkx Graph. If someone
    wants to spend time building a proper data structure, please make a PR, and I will be super
    happy!
    """

    ROOT_ID = 0
    ROOT_HASH = 0

    def __init__(self):
        """Initialize a :class:`_BaseNetworkxTree`."""
        self.data: nx.DiGraph = nx.DiGraph()
        self.data.add_node(self.ROOT_ID, state=None, n_iter=-1)
        self.root_id = self.ROOT_ID
        self.node_names = {self.ROOT_ID: self.ROOT_HASH}
        self.names_to_hash = {self.ROOT_HASH: self.ROOT_ID}
        self._node_count = 0
        self.leafs = {self.ROOT_ID}

    def reset(
        self,
        parent_ids: List[int] = None,
        env_states: States = None,
        model_states: States = None,
        walkers_states: States = None,
    ) -> None:
        """
        Delete all the data currently stored and reset the internal state of \
        the tree .

        Args:
            parent_ids: Ignored. Only to implement interface.
            env_states: Ignored. Only to implement interface.
            model_states: Ignored. Only to implement interface.
            walkers_states: Ignored. Only to implement interface.

        Returns:
            None.
        """
        self.data: nx.DiGraph = nx.DiGraph()
        self.data.add_node(self.ROOT_ID, state=None, n_iter=-1)
        self.root_id = self.ROOT_ID
        self.node_names = {self.ROOT_ID: self.ROOT_HASH}
        self.names_to_hash = {self.ROOT_HASH: self.ROOT_ID}
        self._node_count = 0
        self.leafs = {self.ROOT_ID}

    def get_update_hash(self, node_hash: int) -> None:
        """

        Args:
            node_hash: Unique identifier of a Node.

        Returns:
            None.
        """
        node_name = self.node_names.get(node_hash, None)
        if node_name is None:
            node_name = self.update_hash(node_hash)
        return node_name

    def update_hash(self, node_hash):
        self._node_count += 1
        node_name = int(self._node_count)
        self.node_names[node_hash] = node_name
        self.names_to_hash[node_name] = node_hash
        return node_name

    def append_leaf(
        self,
        leaf_id: int,
        parent_id: int,
        state,
        action,
        dt: int,
        n_iter: int = None,
        from_hash: bool = False,
        reward: float = 0.0,
        cum_reward: float = 0.0,
    ):
        """
        Add a new state as a leaf node of the tree to keep track of the trajectories of the swarm.
        :param leaf_id: Id that identifies the state that will be added to the tree.
        :param parent_id: id that references the state of the system before taking the action.
        :param state: observation assigned to leaf_id state.
        :param action: action taken at leaf_id state.
        :param dt: parameters taken into account when integrating the action.
        :return:
        """
        leaf_name = self.update_hash(leaf_id) if from_hash else leaf_id
        parent_name = self.node_names[parent_id] if from_hash else parent_id
        if leaf_name not in self.data.nodes and leaf_name != parent_name:
            self.data.add_node(
                leaf_name, state=state, n_iter=n_iter, reward=reward, cum_reward=cum_reward
            )
            self.data.add_edge(parent_name, leaf_name, action=action, dt=dt)
            self.leafs.add(leaf_name)

    def prune_tree(self, dead_leafs, alive_leafs, from_hash: bool = False):
        """This prunes the orphan leaves that will no longer be used to save memory."""
        for leaf in dead_leafs:
            self.prune_branch(leaf, alive_leafs, from_hash=from_hash)
        return

    def get_branch(self, leaf_id, from_hash: bool = False, root=ROOT_ID) -> tuple:
        """
        Get the observation from the game ended at leaf_id
        :param leaf_id: id of the leaf node belonging to the branch that will be recovered.
        :return: Sequence of observations belonging to a given branch of the tree.
        """
        leaf_name = self.node_names[leaf_id] if from_hash else leaf_id
        nodes = nx.shortest_path(self.data, root, leaf_name)
        states = [self.data.nodes[n]["state"] for n in nodes]
        actions = [self.data.edges[(n, nodes[i + 1])]["action"] for i, n in enumerate(nodes[:-1])]
        dts = [self.data.edges[(n, nodes[i + 1])]["dt"] for i, n in enumerate(nodes[:-1])]
        return states, actions, dts

    def prune_branch(self, leaf_id, alive_leafs, from_hash: bool = False):
        """This recursively prunes a branch that only leads to an orphan leaf."""
        leaf = self.node_names[leaf_id] if from_hash else leaf_id
        is_not_a_leaf = len(self.data.out_edges([leaf])) > 0
        if is_not_a_leaf:
            self.leafs.discard(leaf)
            return
        elif leaf == self.ROOT_ID or leaf not in self.data.nodes:
            return
        alive_leafs = set([self.node_names[le] if from_hash else le for le in set(alive_leafs)])
        if leaf in alive_leafs:
            return
        parents = set(self.data.in_edges([leaf]))
        self.data.remove_node(leaf)
        self.leafs.discard(leaf)
        for parent, _ in parents:
            return self.prune_branch(parent, alive_leafs)

    def get_parent(self, node_id):
        return list(self.data.in_edges(node_id))[0][0]

    def get_leaf_nodes(self):
        leafs = []
        for node in self.data.nodes:
            if len(self.data.out_edges([node])) == 0:
                leafs.append(node)
        return leafs


class HistoryTree(_BaseNetworkxTree):
    def add_states(
        self,
        parent_ids: List[int],
        env_states: States = None,
        model_states: States = None,
        walkers_states: StatesWalkers = None,
        n_iter: int = None,
    ):
        leaf_ids = walkers_states.id_walkers.tolist()
        for i, (leaf, parent) in enumerate(zip(leaf_ids, parent_ids)):
            state = copy.deepcopy(env_states.states[i])
            reward = copy.deepcopy(env_states.rewards[i])
            cum_reward = copy.deepcopy(walkers_states.cum_rewards[i])
            action = copy.deepcopy(model_states.actions[i])
            dt = copy.copy(model_states.dt[i])
            self.append_leaf(
                leaf,
                parent,
                state,
                action,
                dt,
                n_iter=n_iter,
                from_hash=True,
                reward=reward,
                cum_reward=cum_reward,
            )

    def prune_tree(self, alive_leafs: set, from_hash: bool = False):
        alive_leafs = set([self.node_names[le] if from_hash else le for le in set(alive_leafs)])
        dead_leafs = self.leafs - alive_leafs
        super(HistoryTree, self).prune_tree(
            dead_leafs=dead_leafs, alive_leafs=alive_leafs, from_hash=False
        )
