import numpy as np
import torch
from plangym.env import Environment
from fragile.base_classes import BaseEnvironment
from fragile.states import BaseStates, States
from fragile.utils import to_numpy, device


class DiscreteEnv(BaseEnvironment):
    def __init__(self, env: Environment, device: str = "cpu"):
        self._env = env
        self._n_actions = self._env.action_space.n
        self.device = device

    @property
    def n_actions(self):
        return self._n_actions

    def get_params_dict(self) -> dict:
        params = {
            "states": {
                "sizes": self._env.get_state().shape,
                "dtype": torch.int64,
                "device": self.device,
            },
            "observs": {
                "sizes": self._env.observation_space.shape,
                "dtype": torch.float,
                "device": self.device,
            },
            "rewards": {"sizes": tuple([1]), "dtype": torch.float, "device": self.device},
            "ends": {"sizes": tuple([1]), "dtype": torch.uint8, "device": self.device},
        }
        return params

    # @profile
    def step(self, actions, env_states, n_repeat_action: int = 1, *args, **kwargs) -> BaseStates:
        states = to_numpy(env_states.states)
        actions = to_numpy(actions).astype(np.int32)
        new_states, observs, rewards, ends, infos = self._env.step_batch(
            actions=actions, states=states, n_repeat_action=n_repeat_action
        )

        new_state = self._get_new_states(new_states, observs, rewards, ends, len(actions))
        return new_state

    # @profile
    def reset(self, batch_size: int = 1) -> BaseStates:
        state, obs = self._env.reset()
        states = np.array([state.copy() for _ in range(batch_size)])
        observs = np.array([obs.copy() for _ in range(batch_size)])
        rewards = np.zeros(batch_size, dtype=np.float32)
        ends = np.zeros(batch_size, dtype=np.uint8)
        new_states = self._get_new_states(states, observs, rewards, ends, batch_size)
        return new_states

    # @profile
    def _get_new_states(self, states, observs, rewards, ends, batch_size) -> BaseStates:
        ends = np.array(ends, dtype=np.uint8).reshape(-1, 1)
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        observs = np.array(observs)
        states = np.array(states)
        state = States(state_dict=self.get_params_dict(), n_walkers=batch_size)
        state.update(states=states, observs=observs, rewards=rewards, ends=ends)
        return state
