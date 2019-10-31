from typing import Any, Dict

import numpy as np

from fragile.core.base_classes import StatesOwner
from fragile.core.walkers import States, StatesWalkers


class Critic(StatesOwner):

    STATE_CLASS = States

    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 0

    @classmethod
    def get_params_dict(cls) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def calculate_pest(
        self,
        walkers_states: StatesWalkers = None,
        env_states: States = None,
        model_states: States = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def update(
        self,
        walkers_states: StatesWalkers = None,
        env_states: States = None,
        model_states: States = None,
    ) -> None:
        raise NotImplementedError

    def reset(self, *args, **kwargs) -> None:
        raise NotImplementedError
