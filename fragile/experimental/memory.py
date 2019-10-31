from typing import Union

import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors

from fragile.core.states import States
from fragile.core.utils import fai_iteration_np


class Memory:
    def __init__(self, min_size, max_size, radius: float, *args, **kwargs):
        self.min_size = (min_size,)
        self.max_size = max_size
        self.radius = radius
        self._scores = np.empty(0)
        self._observs = np.empty(0)
        self.kmeans = NearestNeighbors(*args, **kwargs)

    def __len__(self):
        if self._observs is None:
            return 0
        return len(self._observs)

    @property
    def scores(self):
        return self._scores

    @property
    def observs(self):
        return self._observs

    def update(self, states: States = None, observs: Union[np.ndarray, list] = None):
        if states is None and observs is None:
            raise ValueError("Both states and observs cannot be None.")

        if not hasattr(states, "observs") and observs is None:
            raise ValueError(
                "States does not have attribute observs: {} {}".format(states, type(states))
            )
        observs = states.observs if states is not None else observs
        if not isinstance(observs, (np.ndarray, list)):
            raise ValueError(
                "observs must be of type torch.Tensor,"
                " np.ndarray of list,but got {} instead".format(type(observs))
            )

        observs = np.array(observs).copy()

        observs = observs.reshape(len(observs), -1)
        if self.observs is None:
            self._init_memory(observs=observs)
            return

        valid_observs = self._process_scores(observs)
        self._add_to_memory(valid_observs)
        self.kmeans.fit(self.observs)

    def _init_memory(self, observs: np.ndarray):
        self._observs = observs
        self._scores = np.ones(len(observs), dtype=np.float32)
        self.kmeans.fit(self.observs)

    def _process_scores(self, observs: np.ndarray) -> np.ndarray:
        try:
            distances, indices = self.kmeans.kneighbors(observs.reshape(len(observs), -1))
        except sklearn.exceptions.NotFittedError:
            self.kmeans.fit(observs)
            distances, indices = self.kmeans.kneighbors(observs.reshape(len(observs), -1))
        return distances

    def _add_to_memory(self, observs: np.ndarray):
        scores = np.ones(len(observs), dtype=np.float32)
        self._observs = np.concatenate([self.observs, observs])
        self._scores = np.concatenate((self.scores, scores))

    def empty_memory(self):
        compas_ix, will_clone = fai_iteration_np(self.observs, self.scores)
