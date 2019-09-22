from typing import Any, Dict, Optional, Union

import numpy as np

from fragile.core.base_classes import BaseCritic, BaseModel
from fragile.core.env import DiscreteEnv
from fragile.core.states import States
from fragile.core.utils import float_type


class Bounds:
    def __init__(
        self,
        high: Union[np.ndarray, float, int] = np.inf,
        low: Union[np.ndarray, float, int] = -np.inf,
        shape: Optional[tuple] = None,
        dtype: type = None,
    ):

        if shape is None and hasattr(high, "shape"):
            shape = high.shape
        elif shape is None and hasattr(low, "shape"):
            shape = low.shape
        self.shape = shape
        if self.shape is None:
            raise TypeError("If shape is None high or low need to have .shape attribute.")
        self.high = high
        self.low = low
        if dtype is not None:
            self.dtype = dtype
        elif hasattr(high, "dtype"):
            self.dtype = high.dtype
        elif hasattr(low, "dtype"):
            self.dtype = low.dtype
        else:
            self.dtype = type(low)

    def __repr__(self):
        return "{} shape {} dtype {}low {} high {}".format(self.__class__.__name__, self.dtype,
                                                           self.shape, self.low, self.high)

    @classmethod
    def from_tuples(cls, bounds) -> "Bounds":
        low, high = [], []
        for l, h in bounds:
            low.append(l)
            high.append(h)
        low, high = np.array(low), np.array(high)
        return Bounds(low=low, high=high)

    def clip(self, points):
        return np.clip(points, self.low, self.high)

    def points_in_bounds(self, points: np.ndarray) -> np.ndarray:
        return (self.clip(points) == points).all(axis=1).flatten()


class Model(BaseModel):

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


class RandomDiscrete(Model):
    """
    Model that samples actions in a discrete state space using a uniform prior.

    It samples the dt from a normal distribution.
    """

    def __init__(
        self,
        env: DiscreteEnv = None,
        n_actions: int = None,
        dt_sampler: BaseCritic = None,
    ):
        """
        Initialize a :class:`RandomDiscrete`.

        Args:
            env: The number of possible discrete output can be extracted from an Environment.
            n_actions: Number of different discrete. outcomes that the model can provide.
            dt_sampler: dt_sampler used to calculate an additional time step strategy. \
                        the vector output by this class will multiply the actions of the model.
        """
        super(RandomDiscrete, self).__init__(dt_sampler=dt_sampler)
        if n_actions is None and env is None:
            raise ValueError("Env and n_actions cannot be both None.")
        self._n_actions = env.n_actions if n_actions is None else n_actions

    @property
    def n_actions(self):
        """Return the number of different possible discrete actions that the model can output."""
        return self._n_actions

    def get_params_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return the dictionary with the parameters to create a new `RandomDiscrete` model."""
        actions = {"actions": {"dtype": np.int_}}
        return self._add_dt_sample_params(actions)

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
        dt = (1 if self.dt_sampler is None else
              self.dt_sampler.calculate(batch_size=batch_size, model_states=model_states,
                                        **kwargs).astype(int))
        model_states.update(actions=actions, dt=dt)
        return model_states


class RandomContinous(Model):
    """Model that samples actions in a continuous random using a uniform prior."""

    def __init__(self, bounds: Optional[Bounds] = None,
                 low: Optional[Union[int, float, np.ndarray]] = None,
                 high: Optional[Union[int, float, np.ndarray]] = None,
                 shape: Optional[tuple] = None,
                 dt_sampler: Optional[BaseCritic] = None, ):
        """
        Initialize a :class:`RandomContinuous`.

        Args:
            low: Minimum value that the random variable can take.
            high: Maximum value that the random variable can take.
            shape: Shape of the sampled random variable.
            bounds: Bounds class defining the range of allowed values for the model.
        """
        super(RandomContinous, self).__init__(dt_sampler=dt_sampler)
        if shape is not None:
            shape = shape if not isinstance(shape, list) else tuple(shape)

        self.bounds = bounds if bounds is not None else Bounds(low=low, high=high, shape=shape)

    @property
    def shape(self):
        """Return the shape of the sampled random variable."""
        return self.bounds.shape

    @property
    def n_dims(self):
        """Return the number of dimensions of the sampled random variable."""
        return self.bounds.shape[0] if isinstance(self.bounds.shape, tuple) else self.bounds.shape

    def get_params_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return the dictionary with the parameters to create a new `RandomDiscrete` model."""
        actions = {"actions": {"size": self.shape, "dtype": float_type}}
        if self.dt_sampler is not None:
            params = self.dt_sampler.get_params_dict()
            params.update(actions)
        else:
            params = actions
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
        actions = self.random_state.uniform(
            low=self.bounds.low, high=self.bounds.high, size=tuple([batch_size]) + self.shape
        ).astype(self.bounds.dtype)
        dt = (1.0 if self.dt_sampler is None else
              self.dt_sampler.calculate(batch_size=batch_size, model_states=model_states,
                                        **kwargs))
        model_states.update(actions=actions, dt=dt)
        return model_states


class RandomNormal(RandomContinous):

    def __init__(self, bounds: Optional[Bounds] = None,
                 loc: Union[int, float, np.ndarray] = 0.,
                 scale: Optional[Union[int, float, np.ndarray]] = 1.,
                 shape: Optional[tuple] = None,
                 dt_sampler: Optional[BaseCritic] = None):
        """
        Initialize a :class:`RandomContinuous`.

        Args:
            loc: Minimum value that the random variable can take.
            scale: Maximum value that the random variable can take.
            shape: Shape of the sampled random variable.
            bounds: Bounds class defining the range of allowed values for the model.
        """
        super(RandomContinous, self).__init__(dt_sampler=dt_sampler)
        self.loc = loc
        self.scale = scale
        self.bounds = bounds
        if shape is not None:
            shape = shape if not isinstance(shape, list) else tuple(shape)
        self._shape = self.bounds.shape if bounds is not None else shape

    @property
    def shape(self):
        return self._shape

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
        batch_size = batch_size if model_states is None else model_states.n
        actions = self.random_state.normal(size=tuple([batch_size]) + self.shape,
                                           loc=self.loc, scale=self.scale)
        if self.bounds is not None:
            actions = self.bounds.clip(actions).astype(self.bounds.dtype)

        dt = (1.0 if self.dt_sampler is None else
              self.dt_sampler.calculate(batch_size=batch_size, model_states=model_states,
                                        env_states=env_states, walkers_states=walkers_states))
        model_states.update(actions=actions, dt=dt)
        return model_states
