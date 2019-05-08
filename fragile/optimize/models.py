from typing import Tuple

import numpy as np
import torch

from fragile.core.base_classes import BaseStates, BaseEnvironment
from fragile.core.models import RandomContinous
from fragile.core.states import States
from fragile.core.utils import to_tensor, device
from fragile.optimize.env import Function


class UnitaryContinuous(RandomContinous):
    def sample(self, batch_size: int = 1):
        val = super(UnitaryContinuous, self).sample(batch_size=batch_size)
        axis = 1 if len(val.shape) <= 2 else tuple(range(1, len(val.shape)))
        norm = np.linalg.norm(val, axis=axis)
        div = norm.reshape(-1, 1) if axis == 1 else np.expand_dims(np.expand_dims(norm, 1), 1)
        return val / div


class RandomNormal(RandomContinous):
    def __init__(self, env: Function = None, *args, **kwargs):
        kwargs["shape"] = kwargs.get(
            "shape", env.shape if isinstance(env, BaseEnvironment) else None
        )
        super(RandomNormal, self).__init__(env=env, *args, **kwargs)
        self._shape = self.bounds.shape
        self._n_dims = self.bounds.shape

    def sample(self, batch_size: int = 1):
        high = (
            self.bounds.high
            if self.bounds.dtype.kind == "f"
            else self.bounds.high.astype("int64") + 1
        )
        data = np.clip(
            self.np_random.standard_normal(size=tuple([batch_size]) + self.shape).astype(
                self.bounds.dtype
            ),
            self.bounds.low,
            high,
        )
        return to_tensor(data, device=device, dtype=torch.float32)

    def calculate_dt(
        self, model_states: BaseStates, env_states: BaseStates
    ) -> Tuple[np.ndarray, BaseStates]:
        """

        Args:
            model_states:
            env_states:

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.
        """
        dt = np.ones(shape=tuple(env_states.rewards.shape)) * self.mean_dt
        dt = np.clip(dt, self.min_dt, self.max_dt)
        dt = to_tensor(dt, device=device, dtype=torch.float32).reshape(-1, 1)
        model_states.update(dt=dt)
        return dt, model_states

    def reset(self, batch_size: int = 1, *args, **kwargs) -> Tuple[np.ndarray, BaseStates]:
        """

        Args:
            batch_size:
            *args:
            **kwargs:

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.
        """

        model_states = States(state_dict=self.get_params_dict(), n_walkers=batch_size)
        actions = super(RandomNormal, self).sample(batch_size=batch_size)
        model_states.update(dt=np.ones(batch_size), actions=actions, init_actions=actions)
        return actions, model_states
