from typing import Tuple

from gym.spaces import Box
import numpy as np

from fragile.core.base_classes import BaseModel
from fragile.core.env import DiscreteEnv
from fragile.core.states import BaseStates, States


class RandomDiscrete(BaseModel):
    def __init__(
        self,
        env: DiscreteEnv = None,
        n_actions: int = None,
        min_dt=3,
        max_dt=10,
        mean_dt=4,
        std_dt=2,
        *args,
        **kwargs
    ):
        super(RandomDiscrete, self).__init__(*args, **kwargs)
        if n_actions is None and env is None:
            raise ValueError("Env and n_actions cannot be both None.")
        self._n_actions = env.n_actions if n_actions is None else n_actions
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.mean_dt = mean_dt
        self.std_dt = std_dt

    @classmethod
    def get_params_dict(cls) -> dict:
        params = {
            "actions": {"dtype": np.int_},
            "init_actions": {"dtype": np.int_},
            "dt": {"dtype": np.int_},
        }
        return params

    @property
    def n_actions(self):
        return self._n_actions

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
        actions = np.random.randint(0, self.n_actions, size=batch_size)
        model_states.update(dt=np.ones(batch_size), actions=actions, init_actions=actions)
        return actions, model_states

    def predict(
        self,
        env_states: BaseStates = None,
        batch_size: int = None,
        model_states: BaseStates = None,
    ) -> Tuple[np.ndarray, BaseStates]:
        """

        Args:
            env_states:
            batch_size:
            model_states:

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.
        """
        if batch_size is None and env_states is None:
            raise ValueError("env_states and batch_size cannot be both None.")
        size = len(env_states.rewards) if env_states is not None else batch_size
        actions = np.random.randint(0, self.n_actions, size=size)
        return actions, model_states

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
        dt = np.random.normal(
            loc=self.mean_dt, scale=self.std_dt, size=tuple(env_states.rewards.shape)
        )
        dt = np.clip(dt, self.min_dt, self.max_dt).astype(int)
        model_states.update(dt=dt)
        return dt, model_states


class RandomContinous(BaseModel):
    def __init__(self, low, high, env=None, shape=None, min_dt=1, max_dt=10, mean_dt=4, std_dt=1):
        super(RandomContinous, self).__init__()
        if shape is not None:
            shape = shape if not isinstance(shape, list) else tuple(shape)
        self._n_dims = shape
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.mean_dt = mean_dt
        self.std_dt = std_dt
        self.np_random = np.random.RandomState()
        self.bounds = Box(low=low, high=high, shape=shape)

    @property
    def shape(self):
        return self.bounds.shape

    @property
    def n_dims(self):
        return self._n_dims[0] if isinstance(self._n_dims, tuple) else self._n_dims

    def seed(self, seed):
        self.np_random.seed(seed)

    def get_params_dict(self) -> dict:
        params = {
            "actions": {"size": self.shape, "dtype": np.int_},
            "init_actions": {"size": self.shape, "dtype": np.int_},
            "dt": {"size": tuple([self.n_dims]), "dtype": np.int_},
        }
        return params

    def sample(self, batch_size: int = 1):
        high = (
            self.bounds.high
            if self.bounds.dtype.kind == "f"
            else self.bounds.high.astype("int64") + 1
        )
        return self.np_random.uniform(
            low=self.bounds.low, high=high, size=tuple([batch_size]) + self.shape
        ).astype(self.bounds.dtype)

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
        actions = self.sample(batch_size=batch_size)
        model_states.update(dt=np.ones(batch_size), actions=actions, init_actions=actions)
        return actions, model_states

    def predict(
        self,
        env_states: BaseStates = None,
        batch_size: int = None,
        model_states: BaseStates = None,
    ) -> Tuple[np.ndarray, BaseStates]:
        """

        Args:
            env_states:
            batch_size:
            model_states:

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.
        """
        if batch_size is None and env_states is None:
            raise ValueError("env_states and batch_size cannot be both None.")
        size = len(env_states.rewards) if env_states is not None else batch_size
        actions = self.sample(batch_size=size)
        return actions, model_states

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
        dt = np.random.normal(
            loc=self.mean_dt, scale=self.std_dt, size=tuple(env_states.rewards.shape)
        )
        dt = np.clip(dt, self.min_dt, self.max_dt).astype(int)
        model_states.update(dt=dt)
        return dt, model_states
