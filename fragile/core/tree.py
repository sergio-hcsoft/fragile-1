from collections import defaultdict
from typing import List

import numpy as np

from fragile.core.base_classes import States


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


class Tree:
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
        cum_rewards: np.ndarray = None,
    ) -> np.ndarray:
        env_sts = env_states.split_states() if env_states is not None else [None] * len(parent_ids)
        mode_sts = (
            model_states.split_states() if model_states is not None else [None] * len(parent_ids)
        )

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
