from scipy.optimize import minimize

import numpy as np
import torch

from fragile.core.base_classes import BaseStates
from fragile.core.utils import to_numpy, to_tensor
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
            x = to_tensor(x).view(1, -1)
            try:
                y = self.function(x)
            except ZeroDivisionError as e:
                y = -1e7
            return -float(y)

        num_x = to_numpy(x)
        return minimize(_optimize, num_x, bounds=self.bounds, *self.args, **self.kwargs)

    def minimize_point(self, x):
        optim_result = self.minimize(x)
        point = to_tensor(optim_result["x"])
        reward = -1.0 * float(optim_result["fun"])
        return point, reward

    def minimize_batch(self, x: torch.Tensor):
        result = torch.zeros_like(x)
        rewards = torch.zeros((x.shape[0], 1))
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
        actions: [torch.Tensor, np.ndarray],
        env_states: BaseStates,
        n_repeat_action: [int, np.ndarray] = 1,
        *args,
        **kwargs
    ) -> BaseStates:
        """
        Sets the environment to the target states by applying the specified actions an arbitrary
        number of time steps.

        Args:
            actions: Vector containing the actions that will be applied to the target states.
            env_states: BaseStates class containing the state data to be set on the Environment.
            n_repeat_action: Number of times that an action will be applied. If it is an array
                it corresponds to the different dts of each walker.
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            States containing the information that describes the new state of the Environment.
        """
        states = to_tensor(env_states.states, device=self.device)
        actions = to_tensor(actions, device=self.device)
        n_repeat_action = to_tensor(n_repeat_action, device=self.device)
        new_points = actions.float() * n_repeat_action.float() + states.float()

        new_points, rewards = self.minimizer.minimize_batch(new_points)
        ends = self.boundary_condition(new_points, rewards)

        self._last_states = self._get_new_states(new_points, rewards, ends, len(actions))
        return self._last_states
