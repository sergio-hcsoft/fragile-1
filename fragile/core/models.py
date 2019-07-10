from typing import Any, Dict

from gym.spaces import Box
import numpy as np

from fragile.core.base_classes import BaseModel
from fragile.core.env import DiscreteEnv
from fragile.core.states import States


float_type = np.float32


class DtSampler(BaseModel):
    """
    Sample an additional vector of clipped gaussian random variables, and \
    stores it in an attribute called `dt`.
    """

    STATE_CLASS = States

    def __init__(
        self, min_dt: float = 3, max_dt: float = 10, loc_dt: float = 4, scale_dt: float = 2
    ):
        """
        Initialize a :class:`DtSampler`.

        Args:
            min_dt: Minimum dt that will be predicted by the model.
            max_dt: Maximum dt that will be predicted by the model.
            loc_dt: Mean of the gaussian random variable that will model dt.
            scale_dt: Standard deviation of the gaussian random variable that will model dt.

        """
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.mean_dt = loc_dt
        self.std_dt = scale_dt

    @classmethod
    def get_params_dict(cls) -> Dict[str, Dict[str, Any]]:
        """Return the dictionary with the parameters to create a new `RandomDiscrete` model."""
        params = {"dt": {"dtype": float_type}}
        return params

    def sample(
        self,
        batch_size: int,
        model_states: States = None,
        env_states: States = None,
        walkers_states: "StatesWalkers" = None,
    ) -> States:
        """
        Calculate the corresponding data to interact with the Environment and \
        store it in model states.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the environment data.
            env_states: States corresponding to the model data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.

        """
        raise NotImplementedError

    def predict(
        self,
        batch_size: int = None,
        model_states: States = None,
        env_states: States = None,
        walkers_states: "StatesWalkers" = None,
    ) -> States:
        """
        Return States containing the data to interact with the environment and \
        a dt attribute containing clipped gaussian samples.

        Args:
            batch_size: Number of new points to the sampled. If None, env_states.n \
                        will be used to determine the batch_size.
            model_states: States corresponding to the environment data.
            env_states: States corresponding to the model data. Required if \
                        batch_size is None.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Array containing the sampled actions and the new model states variable.

        """
        if batch_size is None and env_states is None:
            raise ValueError("env_states and batch_size cannot be both None.")
        batch_size = batch_size or env_states.n
        model_states = model_states or self.create_new_states(batch_size=batch_size)
        model_states = self.sample(
            batch_size=batch_size,
            model_states=model_states,
            env_states=env_states,
            walkers_states=walkers_states,
        )
        model_states = self.calculate_dt(batch_size=batch_size, model_states=model_states)
        return model_states

    def calculate_dt(self, batch_size: int, model_states: States) -> States:
        """
        Sample the integration step from a clipped gaussian random variable.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the environment data.

        Returns:
            Updated model_states containing an attribute name `dt` with samples \
            from a clipped normal distribution.

        """
        dt = self.random_state.normal(loc=self.mean_dt, scale=self.std_dt, size=batch_size)
        dt = np.clip(dt, self.min_dt, self.max_dt).astype(int)
        model_states.update(dt=dt)
        return model_states

    def reset(self, batch_size: int = 1, model_states: States = None, *args, **kwargs) -> States:
        """
        Return a new blank State for a `RandomDiscrete` instance, and a valid \
        prediction based on that new state.

        Args:
            batch_size: Number of walkers that the new model `State`.
            model_states: States corresponding to the environment data.
            **kwargs: Ignored.

        Returns:
            New model states containing sampled data.

        """
        model_states = self.predict(
            batch_size=batch_size, model_states=model_states, *args, **kwargs
        )
        return model_states


class RandomDiscrete(DtSampler):
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
        loc_dt=4,
        scale_dt=2,
    ):
        """
        Initialize a :class:`RandomDiscrete`.

        Args:
            env: The number of possible discrete output can be extracted from an Environment.
            n_actions: Number of different discrete. outcomes that the model can provide.
            min_dt: Minimum dt that will be predicted by the model.
            max_dt: Maximum dt that will be predicted by the model.
            loc_dt: Mean of the gaussian random variable that will model dt.
            scale_dt: Standard deviation of the gaussian random variable that will model dt.

        """
        super(RandomDiscrete, self).__init__(
            min_dt=min_dt, max_dt=max_dt, loc_dt=loc_dt, scale_dt=scale_dt
        )
        if n_actions is None and env is None:
            raise ValueError("Env and n_actions cannot be both None.")
        self._n_actions = env.n_actions if n_actions is None else n_actions

    @property
    def n_actions(self):
        """Return the number of different possible discrete actions that the model can output."""
        return self._n_actions

    @classmethod
    def get_params_dict(cls) -> Dict[str, Dict[str, Any]]:
        """Return the dictionary with the parameters to create a new `RandomDiscrete` model."""
        actions = {"actions": {"dtype": np.int_}}
        params = super(RandomDiscrete, cls).get_params_dict()
        params.update(actions)
        return params

    def sample(self, batch_size: int, model_states: States = None, **kwargs) -> States:
        """
        Sample a random discrete variable from a uniform prior.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the environment data.

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.

        """
        actions = self.random_state.randint(0, self.n_actions, size=batch_size)
        model_states.update(actions=actions)
        return model_states


class RandomContinous(DtSampler):
    """Model that samples actions in a continuous random using a uniform prior."""

    def __init__(self, low, high, shape=None, min_dt=1, max_dt=10, loc_dt=4, scale_dt=1):
        """
        Initialize a :class:`RandomContinuous`.

        Args:
            low: Minimum value that the random variable can take.
            high: Maximum value that the random variable can take.
            shape: Shape of the sampled random variable.
            min_dt: Minimum dt that will be predicted by the model.
            max_dt: Maximum dt that will be predicted by the model.
            loc_dt: Mean of the gaussian random variable that will model dt.
            scale_dt: Standard deviation of the gaussian random variable that will model dt.
        """
        super(RandomContinous, self).__init__(
            min_dt=min_dt, max_dt=max_dt, loc_dt=loc_dt, scale_dt=scale_dt
        )
        if shape is not None:
            shape = shape if not isinstance(shape, list) else tuple(shape)
        self._n_dims = shape
        self.bounds = Box(low=low, high=high, shape=shape)

    @property
    def shape(self):
        """Return the shape of the sampled random variable."""
        return self.bounds.shape

    @property
    def n_dims(self):
        """Return the number of dimensions of the sampled random variable."""
        return self._n_dims[0] if isinstance(self._n_dims, tuple) else self._n_dims

    def get_params_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return the dictionary with the parameters to create a new `RandomDiscrete` model."""
        params = {
            "actions": {"size": self.shape, "dtype": float_type},
            "dt": {"size": tuple([self.n_dims]), "dtype": float_type},
        }
        return params

    def sample(self, batch_size: int, model_states: States = None, **kwargs) -> States:
        """
        Sample a random continuous variable from a uniform prior.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the model data.

        Returns:
            States containing the new sampled discrete random values.

        """
        high = (
            self.bounds.high
            if self.bounds.dtype.kind == "f"
            else self.bounds.high.astype("int64") + 1
        )
        actions = self.random_state.uniform(
            low=self.bounds.low, high=high, size=tuple([batch_size]) + self.shape
        ).astype(self.bounds.dtype)
        model_states.update(actions=actions)
        return model_states
