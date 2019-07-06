from typing import Tuple

from gym.spaces import Box
import numpy as np

from fragile.core.base_classes import BaseModel
from fragile.core.env import DiscreteEnv
from fragile.core.states import States


class RandomDiscrete(BaseModel):
    """
    Model that samples actions in a discrete state space using a uniform prior.

    It samples the dt from a normal distribution.
    """

    def __init__(
        self,
        env: DiscreteEnv = None,
        n_actions: int = None,
        min_dt=3,
        max_dt=10,
        mean_dt=4,
        std_dt=2,
    ):
        """
        Initialize a :class:`RandomDiscrete`.

        Args:
            env: The number of possible discrete output can be extracted from an Environment.
            n_actions: Number of different discrete. outcomes that the model can provide.
            min_dt: Minimum dt that will be predicted by the model.
            max_dt: Maximum dt that will be predicted by the model.
            mean_dt: Mean of the gaussian random variable that will model dt.
            std_dt: Standard deviation of the gaussian random variable that will model dt.

        """
        super(RandomDiscrete, self).__init__()
        if n_actions is None and env is None:
            raise ValueError("Env and n_actions cannot be both None.")
        self._n_actions = env.n_actions if n_actions is None else n_actions
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.mean_dt = mean_dt
        self.std_dt = std_dt

    @classmethod
    def get_params_dict(cls) -> dict:
        """Return the dictionary with the parameters to create a new `RandomDiscrete` model."""
        params = {
            "actions": {"dtype": np.int_},
            "init_actions": {"dtype": np.int_},
            "dt": {"dtype": np.int_},
        }
        return params

    @property
    def n_actions(self):
        """Return the number of different possible discrete actions that the model can output."""
        return self._n_actions

    def reset(self, batch_size: int = 1, **kwargs) -> Tuple[np.ndarray, States]:
        """
        Return a new blank State for a `RandomDiscrete` instance, and a valid \
        prediction based on that new state.

        Args:
            batch_size: Number of walkers that the new model `State`.
            **kwargs: Ignored.

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.

        """
        model_states = States(state_dict=self.get_params_dict(), batch_size=batch_size)
        actions = np.random.randint(0, self.n_actions, size=batch_size)
        model_states.update(dt=np.ones(batch_size), actions=actions, init_actions=actions)
        return actions, model_states

    def predict(
        self,
        batch_size: int = None,
        model_states: States = None,
        env_states: States = None,
        walkers_states: "StatesWalkers" = None,
    ) -> Tuple[np.ndarray, States]:
        """
        Sample a random discrete variable from a uniform prior.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the environment data.
            env_states: States corresponding to the model data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.

        """
        if batch_size is None and env_states is None:
            raise ValueError("env_states and batch_size cannot be both None.")
        size = len(env_states.rewards) if env_states is not None else batch_size
        actions = np.random.randint(0, self.n_actions, size=size)
        return actions, model_states

    def calculate_dt(
        self,
        model_states: States = None,
        env_states: States = None,
        walkers_states: "StatesWalkers" = None,
    ) -> Tuple[np.ndarray, States]:
        """
        Sample the integration step from a clipped gaussian random variable.

        Args:
            model_states: States corresponding to the environment data.
            env_states: States corresponding to the model data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.

        """
        dt = np.random.normal(loc=self.mean_dt, scale=self.std_dt, size=env_states.n)
        dt = np.clip(dt, self.min_dt, self.max_dt).astype(int)
        model_states.update(dt=dt)
        return dt, model_states


class RandomContinous(BaseModel):
    """Model that samples actions in a continuous random using a uniform prior."""

    def __init__(self, low, high, shape=None, min_dt=1, max_dt=10, mean_dt=4, std_dt=1):
        """
        Initialize a :class:`RandomContinuous`.

        Args:
            low: Minimum value that the random variable can take.
            high: Maximum value that the random variable can take.
            shape: Shape of the sampled random variable.
            min_dt: Minimum dt that will be predicted by the model.
            max_dt: Maximum dt that will be predicted by the model.
            mean_dt: Mean of the gaussian random variable that will model dt.
            std_dt: Standard deviation of the gaussian random variable that will model dt.
        """
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
        """Return the shape of the sampled random variable."""
        return self.bounds.shape

    @property
    def n_dims(self):
        """Return the number of dimensions of the sampled random variable."""
        return self._n_dims[0] if isinstance(self._n_dims, tuple) else self._n_dims

    def seed(self, seed):
        """Set the random seed of the random number generator."""
        self.np_random.seed(seed)

    def get_params_dict(self) -> dict:
        """Return an state_dict to be used for instantiating an States class."""
        params = {
            "actions": {"size": self.shape, "dtype": np.int_},
            "init_actions": {"size": self.shape, "dtype": np.int_},
            "dt": {"size": tuple([self.n_dims]), "dtype": np.int_},
        }
        return params

    def sample(self, batch_size: int = 1, **kwargs):
        """Sample a random continuous variable from a uniform prior."""
        high = (
            self.bounds.high
            if self.bounds.dtype.kind == "f"
            else self.bounds.high.astype("int64") + 1
        )
        return self.np_random.uniform(
            low=self.bounds.low, high=high, size=tuple([batch_size]) + self.shape
        ).astype(self.bounds.dtype)

    def reset(self, batch_size: int = 1, *args, **kwargs) -> Tuple[np.ndarray, States]:
        """
        Return a new blank State for a `RandomDiscrete` instance, and a valid \
        prediction based on that new state.

        Args:
            batch_size: Number of new points to the sampled.
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.

        """
        model_states = States(state_dict=self.get_params_dict(), batch_size=batch_size)
        actions = self.sample(batch_size=batch_size)
        model_states.update(dt=np.ones(batch_size), actions=actions, init_actions=actions)
        return actions, model_states

    def predict(
        self,
        env_states: States = None,
        batch_size: int = None,
        model_states: States = None,
        walkers_states: "StatesWalkers" = None,
    ) -> Tuple[np.ndarray, States]:
        """
        Sample a random continuous variable from a uniform prior.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the environment data.
            env_states: States corresponding to the model data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.

        """
        if batch_size is None and env_states is None:
            raise ValueError("env_states and batch_size cannot be both None.")
        size = len(env_states.rewards) if env_states is not None else batch_size
        actions = self.sample(batch_size=size)
        return actions, model_states

    def calculate_dt(
        self,
        model_states: States = None,
        env_states: States = None,
        batch_size: int = None,
        walkers_states: "StatesWalkers" = None,
    ) -> Tuple[np.ndarray, States]:
        """
        Sample the integration step from a clipped gaussian random variable.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the environment data.
            env_states: States corresponding to the model data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.

        """
        dt = np.random.normal(loc=self.mean_dt, scale=self.std_dt, size=env_states.n)
        dt = np.clip(dt, self.min_dt, self.max_dt).astype(int)
        model_states.update(dt=dt)
        return dt, model_states
