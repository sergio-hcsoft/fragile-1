import torch
import numpy as np
from typing import Tuple
from fragile.states import BaseStates, States
from fragile.base_classes import BaseModel


class RandomDiscrete(BaseModel):
    def __init__(self, n_actions: int, min_dt=1, max_dt=10, mean_dt=4, std_dt=1, *args, **kwargs):
        super(RandomDiscrete, self).__init__(*args, **kwargs)
        self._n_actions = n_actions
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.mean_dt = mean_dt
        self.std_dt = std_dt

    @classmethod
    def get_params_dict(cls) -> dict:
        params = {
            "actions": {"sizes": tuple([1]), "dtype": torch.int},
            "init_actions": {"sizes": tuple([1]), "dtype": torch.int},
            "dt": {"sizes": tuple([1]), "dtype": torch.int},
        }
        return params

    @property
    def n_actions(self):
        return self._n_actions

    def reset(self, batch_size: int = 1, *args, **kwargs) -> Tuple[np.ndarray, BaseStates]:

        model_states = States(state_dict=self.get_params_dict(), n_walkers=batch_size)
        actions = np.random.randint(0, self.n_actions, size=batch_size)
        model_states.update(dt=np.ones(batch_size), actions=actions, init_actions=actions)
        return actions, model_states

    def predict(self, env_states=None, batch_size: int = None, model_states=None) -> Tuple:
        if batch_size is None and env_states is None:
            raise ValueError("env_states and batch_size cannot be both None.")
        size = len(env_states.rewards) if env_states is not None else batch_size
        actions = np.random.randint(0, self.n_actions, size=size)
        return actions, model_states

    def calculate_dt(self, model_states, env_states) -> Tuple:
        dt = np.random.normal(
            loc=self.mean_dt, scale=self.std_dt, size=tuple(env_states.rewards.shape)
        )
        dt = np.clip(dt, self.min_dt, self.max_dt).astype(int)
        model_states.update(dt=dt)
        return dt, model_states
