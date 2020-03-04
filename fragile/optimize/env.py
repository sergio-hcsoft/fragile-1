from typing import Callable, Union

import numpy as np

from scipy.optimize import minimize
from scipy.optimize import Bounds as ScipyBounds

from fragile.core.env import Environment
from fragile.core.states import StatesEnv, StatesModel, StatesWalkers
from fragile.core.models import Bounds


class Function(Environment):
    """
    Environment that represents an arbitrary mathematical function.
    """

    def __init__(
        self, function: Callable, bounds: Bounds,
    ):
        if not isinstance(bounds, Bounds):
            raise TypeError("Bounds needs to be an instance of Bounds, found {}".format(bounds))
        self.function = function
        self.bounds = bounds
        self.shape = self.bounds.shape
        super(Function, self).__init__(observs_shape=self.shape, states_shape=self.shape)

    @classmethod
    def from_bounds_params(
        cls,
        function: Callable,
        shape: tuple = None,
        high: Union[int, float, np.ndarray] = np.inf,
        low: Union[int, float, np.ndarray] = -np.inf,
    ) -> "Function":
        if (
            not isinstance(high, np.ndarray)
            and not isinstance(low, np.ndarray) is None
            and shape is None
        ):
            raise TypeError("Need to specify shape or high or low must be a numpy array.")
        bounds = Bounds(high=high, low=low, shape=shape)
        return Function(function=function, bounds=bounds)

    @property
    def func(self) -> Callable:
        return self.function

    def __repr__(self):
        text = "{} with function {}, obs shape {}, and bounds: {}".format(
            self.__class__.__name__, self.func.__name__, self.shape, self.bounds
        )
        return text

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """
        Sets the environment to the target states by applying the specified actions an arbitrary
        number of time steps.

        Args:
            model_states: States corresponding to the model data.
            env_states: States class containing the state data to be set on the Environment.

        Returns:
            States containing the information that describes the new state of the Environment.
        """
        new_points = model_states.actions + env_states.observs
        ends = self.calculate_end(points=new_points)
        rewards = self.function(new_points).flatten()

        updated_states = self.states_from_data(
            states=new_points,
            observs=new_points,
            rewards=rewards,
            ends=ends,
            batch_size=model_states.n,
        )
        return updated_states

    def reset(self, batch_size: int = 1, **kwargs) -> StatesEnv:
        """
        Resets the environment to the start of a new episode and returns an
        States instance describing the state of the Environment.
        Args:
            batch_size: Number of walkers that the returned state will have.
            **kwargs: Ignored. This environment resets without using any external data.

        Returns:
            States instance describing the state of the Environment. The first
            dimension of the data tensors (number of walkers) will be equal to
            batch_size.
        """
        ends = np.zeros(batch_size, dtype=np.bool_)
        new_points = self.sample_bounds(batch_size=batch_size)
        rewards = self.function(new_points).flatten()
        new_states = self.states_from_data(
            states=new_points,
            observs=new_points,
            rewards=rewards,
            ends=ends,
            batch_size=batch_size,
        )
        return new_states

    def calculate_end(self, points):
        return np.logical_not(self.bounds.points_in_bounds(points)).flatten()

    def sample_bounds(self, batch_size: int):
        new_points = np.zeros(tuple([batch_size]) + self.shape, dtype=np.float32)
        for i in range(batch_size):
            new_points[i, :] = self.random_state.uniform(
                low=self.bounds.low, high=self.bounds.high, size=self.shape
            )
        return new_points


class Minimizer:
    def __init__(self, function: Function, bounds=None, *args, **kwargs):
        self.env = function
        self.function = function.function
        self.bounds = self.env.bounds if bounds is None else bounds
        self.args = args
        self.kwargs = kwargs

    def minimize(self, x):
        def _optimize(x):
            try:
                x = x.reshape((1,) + x.shape)
                y = self.function(x)
            except (ZeroDivisionError, RuntimeError) as e:
                y = np.inf
            return y

        bounds = ScipyBounds(
            ub=self.bounds.high if self.bounds is not None else None,
            lb=self.bounds.low if self.bounds is not None else None,
        )
        return minimize(_optimize, x, bounds=bounds, *self.args, **self.kwargs)

    def minimize_point(self, x):
        optim_result = self.minimize(x)
        point = optim_result["x"]
        reward = float(optim_result["fun"])
        return point, reward

    def minimize_batch(self, x: np.ndarray):
        result = np.zeros_like(x)
        rewards = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            new_x, reward = self.minimize_point(x[i, :])
            result[i, :] = new_x
            rewards[i, :] = float(reward)
        return result, rewards


class MinimizerWrapper(Function):
    def __init__(self, function: Function, *args, **kwargs):
        self.env = function
        self.minimizer = Minimizer(function=self.env, *args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.env, item)

    def __repr__(self):
        return self.env.__repr__()

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """
        Sets the environment to the target states by applying the specified actions an arbitrary
        number of time steps.

        Args:

            env_states: States class containing the state data to be set on the Environment.
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            States containing the information that describes the new state of the Environment.
        """
        env_states = super(MinimizerWrapper, self).step(
            model_states=model_states, env_states=env_states
        )

        new_points, rewards = self.minimizer.minimize_batch(env_states.observs)
        ends = np.logical_not(self.bounds.points_in_bounds(new_points)).flatten()
        optim_states = self._get_new_states(new_points, rewards.flatten(), ends, model_states.n)
        return optim_states
