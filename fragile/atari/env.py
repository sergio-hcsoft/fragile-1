import copy

import numpy

from fragile.core.env import DiscreteEnv
from fragile.core.states import StatesEnv, StatesModel


class AtariEnv(DiscreteEnv):
    """The AtariEnv acts as an interface with `plangym.AtariEnvironment`.

    It can interact with any Atari environment that follows the interface of ``plangym``.
    """

    STATE_CLASS = StatesEnv

    def step(self, model_states: StatesModel, env_states: StatesEnv) -> StatesEnv:
        """
        Set the environment to the target states by applying the specified \
        actions an arbitrary number of time steps.

        Args:
            model_states: States representing the data to be used to act on the environment.
            env_states: States representing the data to be set in the environment.

        Returns:
            States containing the information that describes the new state of the Environment.

        """
        actions = model_states.actions.astype(numpy.int32)
        n_repeat_actions = model_states.dt if hasattr(model_states, "dt") else 1
        new_states, observs, rewards, ends, infos = self._env.step_batch(
            actions=actions, states=env_states.states, n_repeat_action=n_repeat_actions
        )
        game_ends = [inf.get("game_end", False) for inf in infos]

        new_state = self.states_from_data(
            states=new_states,
            observs=observs,
            rewards=rewards,
            oobs=ends,
            batch_size=len(actions),
            terminals=game_ends,
        )
        return new_state

    def reset(self, batch_size: int = 1, **kwargs) -> StatesEnv:
        """
        Reset the environment to the start of a new episode and returns a new \
        :class:`StatesEnv` instance describing the state of the :class:`AtariEnvironment`.

        Args:
            batch_size: Number of walkers of the returned state.
            **kwargs: Ignored. This environment resets without using any external data.

        Returns:
            :class:`StatesEnv` instance describing the state of the Environment. \
            The first dimension of the data tensors (number of walkers) will be \
            equal to batch_size.

        """
        state, obs = self._env.reset()
        states = numpy.array([copy.deepcopy(state) for _ in range(batch_size)])
        observs = numpy.array([copy.deepcopy(obs) for _ in range(batch_size)])
        rewards = numpy.zeros(batch_size, dtype=numpy.float32)
        ends = numpy.zeros(batch_size, dtype=numpy.bool_)
        game_ends = numpy.zeros(batch_size, dtype=numpy.bool_)
        new_states = self.states_from_data(
            states=states,
            observs=observs,
            rewards=rewards,
            oobs=ends,
            batch_size=batch_size,
            terminals=game_ends,
        )
        return new_states
