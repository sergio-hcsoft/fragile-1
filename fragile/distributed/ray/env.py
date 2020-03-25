from typing import Callable, Dict

import numpy

from fragile.core.env import BaseEnvironment, Environment
from fragile.core.states import StatesEnv, StatesModel
from fragile.core.utils import StateDict
from fragile.distributed.ray import ray


@ray.remote
class Environment(Environment):
    """
    :class:`fragile.Environment` remote interface to be used with ray.

    Wraps a :class:`fragile.Environment` passed as a callable.
    """

    def __init__(self, env_callable: Callable[[dict], Environment], env_kwargs: dict = None):
        """
        Initialize a :class:`Environment`.

        Args:
            env_callable: Callable that returns a :class:`fragile.Environment`.
            env_kwargs: Passed to ``env_callable``.

        """
        env_kwargs = {} if env_kwargs is None else env_kwargs
        self.env = env_callable(**env_kwargs)

    def __getattr__(self, item):
        return getattr(self.env, item)

    def get(self, name: str, default=None):
        """
        Get an attribute from the wrapped environment.

        Args:
            name: Name of the target attribute.

        Returns:
            Attribute from the wrapped :class:`fragile.Environment`.

        """
        try:
            return getattr(self.env, name)
        except Exception:
            return default

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """
        Step the wrapped :class:`fragile.Environment`.

        Args:
            model_states: States representing the data to be used to act on the environment.
            env_states: States representing the data to be set in the environment.

        Returns:
            States representing the next state of the environment and all \
            the needed information.

        """
        return BaseEnvironment.step(self, model_states=model_states, env_states=env_states)

    def states_from_data(self, batch_size: int, **kwargs) -> StatesEnv:
        """
        Initialize a :class:`StatesEnv` with the data provided as kwargs.

        Args:
            batch_size: Number of elements in the first dimension of the \
                       :class:`State` attributes.
            **kwargs: Attributes that will be added to the returned :class:`States`.

        Returns:
            A new :class:`StatesEmv` created with the ``params_dict``, and \
            updated with the attributes passed as keyword arguments.

        """
        return self.env.states_from_data(batch_size=batch_size, **kwargs)

    def make_transitions(self, *args, **kwargs) -> Dict[str, numpy.ndarray]:
        """
        Return the data corresponding to the new state of the environment after \
        using the input data to make the corresponding state transition.

        Args:
            *args: List of arguments passed if the returned value from the \
                  ``states_to_data`` function of the class was a tuple.
            **kwargs: Keyword arguments passed if the returned value from the \
                  ``states_to_data`` function of the class was a dictionary.

        Returns:
            Dictionary containing the data representing the state of the environment \
            after the state transition. The keys of the dictionary are the names of \
            the data attributes and its values are arrays representing a batch of \
            new values for that attribute.

            The :class:`StatesEnv` returned by ``step`` will contain the returned \
            data.

        """
        return self.env.make_transitions(*args, **kwargs)

    def reset(
        self, batch_size: int = 1, env_states: StatesEnv = None, *args, **kwargs
    ) -> StatesEnv:
        """
        Reset the wrapped :class:`fragile.Environment` and return an States class \
        with batch_size copies of the initial state.

        Args:
           batch_size: Number of walkers that the resulting state will have.
           env_states: States class used to set the environment to an arbitrary \
                       state.
           args: Additional arguments not related to environment data.
           kwargs: Additional keyword arguments not related to environment data.

        Returns:
           States class containing the information of the environment after the \
            reset.

        """
        return self.env.reset(batch_size=batch_size, env_states=env_states, *args, **kwargs)

    def get_params_dict(self) -> StateDict:
        """Return the parameter dictionary of the wrapped :class:`fragile.Environment`."""
        return self.env.get_params_dict()
