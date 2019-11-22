from collections import defaultdict
import copy
from typing import List

import networkx as nx
import numpy as np

from fragile.core.base_classes import States, BaseStateTree
from fragile.core.walkers import StatesWalkers


class Node:
    def __init__(
        self,
        node_id: int,
        parent_id: int,
        env_state: States,
        model_state: States,
        reward: float = None,
    ):
        self._node_id = node_id
        self._parent_id = parent_id
        self.env_state = env_state
        self.model_state = model_state
        self.reward = reward

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def parent_id(self) -> int:
        return self._parent_id

    @property
    def end(self) -> bool:
        return bool(self.env_state.ends)


class Tree(BaseStateTree):
    def __init__(self):
        self._curr_id = 0
        self.parents = {}
        self.nodes = {}
        self.children = defaultdict(list)
        self._id_generator = self._new_id_generator()

    @property
    def curr_id(self):
        return self._curr_id

    def has_children(self, node_id):
        return len(self.children[node_id]) > 0

    def _new_id_generator(self):
        self._curr_id = 0
        while True:
            yield self._curr_id
            self._curr_id += 1

    def new_id(self):
        return next(self._id_generator)

    def add_one(self, parent_id, env_state, model_state, reward) -> int:
        node_id = self.new_id()
        # print("node_id : {} parent_id: {}".format(node_id, parent_id))
        node = Node(
            node_id=node_id,
            parent_id=parent_id,
            env_state=env_state.copy(),
            model_state=model_state.copy(),
            reward=reward,
        )
        self.nodes[node_id] = node
        self.children[parent_id].append(node_id)
        self.parents[node_id] = parent_id
        return node_id

    def add_states(
        self,
        parent_ids: List[int],
        env_states: States = None,
        model_states: States = None,
        walkers_states: np.ndarray = None,
    ) -> np.ndarray:
        env_sts = env_states.split_states() if env_states is not None else [None] * len(parent_ids)
        mode_sts = (
            model_states.split_states() if model_states is not None else [None] * len(parent_ids)
        )
        cum_rewards = copy.deepcopy(walkers_states.cum_rewards)
        node_ids = []
        for i, (parent_id, env_state, model_state) in enumerate(
            zip(parent_ids, env_sts, mode_sts)
        ):
            node_id = self.add_one(
                parent_id=parent_id,
                reward=cum_rewards[i],
                env_state=env_state,
                model_state=model_state,
            )
            node_ids.append(node_id)
        return np.array(node_ids, dtype=int)

    def reset(self, env_state: States, model_state: States, reward=None):
        if isinstance(env_state, States):
            if env_state.n > 1:
                env_state = list(env_state.split_states())[0]

        if isinstance(model_state, States):
            if model_state.n > 1:
                model_state = list(model_state.split_states())[0]

        self.parents = {}
        self.nodes = {}
        self.children = defaultdict(list)
        self._id_generator = self._new_id_generator()

        self.add_one(parent_id=None, env_state=env_state, model_state=model_state, reward=reward)

    def get_path_ids(self, end_node: int):
        nodes = []
        node_id = int(end_node)
        while node_id is not None:
            # node = self.nodes[node_id]
            nodes.append(node_id)
            node_id = self.parents[node_id]
        return nodes[::-1]

    def prune_branch(self, leaf_id):
        if self.has_children(leaf_id):
            raise ValueError(
                "You cannot delete a node that has children. Node id: {}".format(leaf_id)
            )
        while not self.has_children(leaf_id) and self.parents[leaf_id] > 0:
            new_id = int(self.parents[leaf_id])
            del self.nodes[leaf_id]
            del self.parents[leaf_id]
            del self.children[leaf_id]
            leaf_id = new_id


class BaseNetworkxTree(BaseStateTree):
    """This is a tree data structure that stores the paths followed by the walkers. It can be
    pruned to delete paths that will no longer be needed. It uses a networkx Graph. If someone
    wants to spend time building a proper data structure, please make a PR, and I will be super
    happy!
    """

    def __init__(self):
        self.data: nx.DiGraph = nx.DiGraph()
        self.data.add_node(0, state=None)
        self.root_id = 0
        self.node_names = {0: 0}
        self.names_to_hash = {0: 0}
        self._node_count = 0
        self.leafs = {0}

    def reset(self, parent_ids: List[int] = None, env_states: States = None,
              model_states: States = None, walkers_states: States = None) -> None:
        self.data: nx.DiGraph = nx.DiGraph()
        self.data.add_node(0, state=None)
        self.root_id = 0
        self.node_names = {0: 0}
        self._node_count = 0
        self.leafs = {0}

    def get_update_hash(self, node_hash):
        node_name = self.node_names.get(node_hash, None)
        if node_name is None:
            self._node_count += 1
            node_name = self._node_count
            self.node_names[node_hash] = node_name
            self.names_to_hash[node_name] = node_hash
        return node_name

    def append_leaf(self, leaf_id: int, parent_id: int, state, action, dt: int,
                    from_hash: bool = False):
        """
        Add a new state as a leaf node of the tree to keep track of the trajectories of the swarm.
        :param leaf_id: Id that identifies the state that will be added to the tree.
        :param parent_id: id that references the state of the system before taking the action.
        :param state: observation assigned to leaf_id state.
        :param action: action taken at leaf_id state.
        :param dt: parameters taken into account when integrating the action.
        :return:
        """
        leaf_name = self.get_update_hash(leaf_id) if from_hash else leaf_id
        parent_name = self.get_update_hash(parent_id) if from_hash else parent_id
        if leaf_name not in self.data.nodes and leaf_name != parent_name:
            self.data.add_node(leaf_name, state=state)
            self.data.add_edge(parent_name, leaf_name, action=action, dt=dt)
            self.leafs.add(leaf_name)

    def prune_tree(self, dead_leafs, alive_leafs, from_hash:bool=False):
        """This prunes the orphan leaves that will no longer be used to save memory."""
        for leaf in dead_leafs:
            self.prune_branch(leaf, alive_leafs, from_hash=from_hash)
        return

    def get_branch(self, leaf_id, from_hash: bool = False) -> tuple:
        """
        Get the observation from the game ended at leaf_id
        :param leaf_id: id of the leaf node belonging to the branch that will be recovered.
        :return: Sequence of observations belonging to a given branch of the tree.
        """
        leaf_name = self.node_names[leaf_id] if from_hash else leaf_id
        nodes = nx.shortest_path(self.data, 0, leaf_name)
        states = [self.data.nodes[n]["state"] for n in nodes]
        actions = [self.data.edges[(n, nodes[i+1])]["action"] for i, n in enumerate(nodes[:-1])]
        dts = [self.data.edges[(n, nodes[i + 1])]["dt"] for i, n in enumerate(nodes[:-1])]
        return states, actions, dts

    def prune_branch(self, leaf_id, alive_leafs, from_hash: bool = False):
        """This recursively prunes a branch that only leads to an orphan leaf."""
        leaf = self.node_names[leaf_id] if from_hash else leaf_id
        is_not_a_leaf = len(self.data.out_edges([leaf])) > 0
        if is_not_a_leaf:
            self.leafs.discard(leaf)
            return
        elif leaf == 0 or leaf not in self.data.nodes:
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


class HistoryTree(BaseNetworkxTree):

    def add_states(
        self,
        parent_ids: List[int],
        env_states: States = None,
        model_states: States = None,
        walkers_states: StatesWalkers = None,
    ) -> np.ndarray:
        leaf_ids = walkers_states.id_walkers.tolist()
        for i, (leaf, parent) in enumerate(zip(leaf_ids, parent_ids)):
            state = env_states.states[i].copy()
            action = copy.deepcopy(model_states.actions[i])
            dt = copy.copy(model_states.dt[i])
            self.append_leaf(leaf, parent, state, action, dt, from_hash=True)

    def prune_tree(self,  alive_leafs: set, from_hash: bool=False):
        alive_leafs = set([self.node_names[le] if from_hash else le for le in set(alive_leafs)])
        dead_leafs = self.leafs - alive_leafs
        super(HistoryTree, self).prune_tree(dead_leafs=dead_leafs, alive_leafs=alive_leafs,
                                            from_hash=False)