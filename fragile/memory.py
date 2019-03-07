import torch
import numpy as np
from fragile.states import States
from sklearn.neighbors import NearestNeighbors
from fragile.utils import device
from fragile.utils import relativize_np


class Memory:
    def __init__(self, min_size, max_size, radius: float, *args, **kwargs):
        self.min_size = (min_size,)
        self.max_size = max_size
        self.radius = radius
        self._score = None
        self._observs = None
        self.kmeans = NearestNeighbors(*args, **kwargs)

    def __len__(self):
        if self._observs is None:
            return 0
        return len(self._observs)

    def score(self):
        return self._score

    @property
    def observs(self):
        return self._observs

    def update(self, states: States = None, observs: [torch.Tensor, np.ndarray, list] = None):
        if states is None and observs is None:
            raise ValueError("Both states and observs cannot be None.")

        if not hasattr(states, "observs") and observs is None:
            raise ValueError(
                "States does not have attribute observs: {} {}".format(states, type(states))
            )
        observs = states.observs if states is not None else observs
        if not isinstance(observs, (np.ndarray, list, torch.Tensor)):
            raise ValueError(
                "observs must be of type torch.Tensor, np.ndarray of list,but got {} instead".format(
                    type(observs)
                )
            )

        observs = (
            observs.cpu().numpy().copy()
            if isinstance(observs, torch.Tensor)
            else np.array(observs).copy()
        )
        observs = observs.reshape(len(observs), -1)
        if self.observs is None:
            self._init_memory(observs=observs)
            return

        valid_observs, _ = self._process_scores(observs)
        self._add_to_memory(valid_observs)
        self.kmeans.fit(self.observs)

    def _init_memory(self, observs: np.ndarray):
        self._observs = observs
        self._score = np.ones(len(observs), dtype=np.float32)
        self.kmeans.fit(self.observs)

    def _process_scores(self, observs: np.ndarray) -> np.ndarray:
        distances, indices = self.kmeans.kneighbors(observs.reshape(len(observs), -1))
        return distances

    def _add_to_memory(self, observs: np.ndarray):
        scores = np.ones(len(observs), dtype=np.float32)
        return observs
