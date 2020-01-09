import numpy as np
from scipy.optimize import minimize

from fragile.core.base_classes import States
from fragile.optimize.env import Function


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
                y = -float(self.function(x))
            except (ZeroDivisionError, RuntimeError) as e:
                y = np.inf
            return y

        return minimize(_optimize, x, bounds=self.bounds, *self.args, **self.kwargs)

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

    def step(
        self,
        actions: np.ndarray,
        env_states: States,
        n_repeat_action: [int, np.ndarray] = 1,
        *args,
        **kwargs
    ) -> States:
        """
        Sets the environment to the target states by applying the specified actions an arbitrary
        number of time steps.

        Args:
            actions: Vector containing the actions that will be applied to the target states.
            env_states: States class containing the state data to be set on the Environment.
            n_repeat_action: Number of times that an action will be applied. If it is an array
                it corresponds to the different dts of each walker.
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            States containing the information that describes the new state of the Environment.
        """
        states = env_states.states
        new_points = actions * n_repeat_action + states

        new_points, rewards = self.minimizer.minimize_batch(new_points)

        ends = self.boundary_condition(new_points, rewards)

        self._last_states = self._get_new_states(new_points, rewards, ends, len(actions))
        return self._last_states
