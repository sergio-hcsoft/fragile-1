from typing import Callable

from fragile.core.models import Bounds, ContinuousUniform
from fragile.core.swarm import Swarm
from fragile.core.states import StatesEnv, StatesModel, StatesWalkers
from fragile.core.walkers import Walkers
from fragile.optimize.env import Function


class FunctionMapper(Swarm):
    """It is a swarm adapted to minimize mathematical functions."""

    def __init__(
        self,
        walkers: Callable = Walkers,
        model: Callable = ContinuousUniform,
        accumulate_rewards: bool = False,
        minimize: bool = True,
        start_same_pos: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize a :class:`FunctionMapper`.

        Args:
            model: A function that returns an instance of a :class:`Model`.
            walkers: A function that returns an instance of :class:`Walkers`.
            accumulate_rewards: If ``True`` the rewards obtained after transitioning \
                                to a new state will accumulate. If ``False`` only the last \
                                reward will be taken into account.
            minimize: If ``True`` the algorithm will perform a minimization \
                      process. If ``False`` it will be a maximization process.
            start_same_pos: If ``True`` all the walkers will have the same \
                            starting position.
            *args: Passed :class:`Walkers` __init__.
            **kwargs: Passed :class:`Walkers` __init__.
        """
        super(FunctionMapper, self).__init__(
            walkers=walkers,
            model=model,
            accumulate_rewards=accumulate_rewards,
            minimize=minimize,
            *args,
            **kwargs
        )
        self.start_same_pos = start_same_pos

    @classmethod
    def from_function(
        cls, function: Callable, bounds: Bounds, *args, **kwargs
    ) -> "FunctionMapper":
        """
        Initialize a :class:`FunctionMapper` using a python callable and a \
        :class:`Bounds` instance.

        Args:
            function: Callable representing an arbitrary function to be optimized.
            bounds: Represents the domain of the function to be optimized.
            *args: Passed to :class:`FunctionMapper` __init__.
            **kwargs: Passed to :class:`FunctionMapper` __init__.

        Returns:
            Instance of :class:`FunctionMapper` that optimizes the target function.
        """
        env = Function(function=function, bounds=bounds,)
        return FunctionMapper(env=lambda: env, *args, **kwargs)

    def __repr__(self):
        return "{}\n{}".format(self.env.__repr__(), super(FunctionMapper, self).__repr__())

    def reset(
        self,
        walkers_states: StatesWalkers = None,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
    ):
        """
        Reset the :class:`fragile.Walkers`, the :class:`Function` environment, the \
        :class:`Model` and clear the internal data to start a new search process.

        Args:
            model_states: :class:`StatesModel` that define the initial state of \
                          the :class:`Model`.
            env_states: :class:`StatesEnv` that define the initial state of \
                        the :class:`Function`.
            walkers_states: :class:`StatesWalkers` that define the internal \
                            states of the :class:`Walkers`.

        """
        super(FunctionMapper, self).reset(
            walkers_states=walkers_states, model_states=model_states, env_states=env_states
        )
        if self.start_same_pos:
            self.walkers.env_states.observs[:] = self.walkers.env_states.observs[0]
            self.walkers.env_states.states[:] = self.walkers.env_states.states[0]
