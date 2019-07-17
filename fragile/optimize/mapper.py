import copy
from typing import Callable

import numpy as np

from fragile.core.models import RandomContinous
from fragile.core.states import States
from fragile.core.swarm import Swarm
from fragile.core.utils import update_defaults
from fragile.core.walkers import float_type, Walkers
from fragile.optimize.encoder import Critic
from fragile.optimize.env import Function


class MapperWalkers(Walkers):
    def __init__(self, encoder: Critic = None, minimize:bool=True,  *args, **kwargs):
        """
        Initialize a :class:`MapperWalkers`.

        Args:
            encoder: Encoder that will be used to calculate the pests.
            *args:
            **kwargs:
        """
        # Add data specific to the child class in the StatesWalkers class as new attributes.
        pests = np.zeros(kwargs["n_walkers"], dtype=float_type)
        super(MapperWalkers, self).__init__(
            pests=pests, best_reward_found=-1e10, best_found=None, *args, **kwargs
        )
        self.critic = encoder
        self.minimize = minimize

    def __repr__(self):
        text = "Best reward found: {:.5f} at position: {}," "Encoder: \n {}".format(
            float(self.states.best_reward_found), self.states.best_found, self.critic
        )
        return text + super(MapperWalkers, self).__repr__()

    def calculate_virtual_reward(self):
        super(MapperWalkers, self).calculate_virtual_reward()
        if self.critic is not None:
            self.critic.calculate_pest(
                walkers_states=self.states,
                model_states=self.model_states,
                env_states=self.env_states,
            )
            virt_rew = self.states.virtual_rewards * self.states.pests
        else:
            virt_rew = self.states.virtual_rewards
        self.states.update(virtual_rewards=virt_rew)

    def balance(self):
        self.update_best()
        returned = super(MapperWalkers, self).balance()
        if self.critic is not None:
            self.critic.update(
                walkers_states=self.states,
                model_states=self.model_states,
                env_states=self.env_states,
            )
        return returned

    def update_best(self):
        ix = self.states.cum_rewards.argmax()
        best = self.env_states.observs[ix].copy()
        best_reward = float(self.states.cum_rewards[ix])
        best_is_alive = not bool(self.env_states.ends[ix])
        if self.states.best_reward_found < best_reward and best_is_alive:
            self.states.update(best_reward_found=best_reward)
            self.states.update(best_found=best)

    def reset(self, env_states: States = None, model_states: States = None):
        super(MapperWalkers, self).reset(env_states=env_states, model_states=model_states)
        ix = self.states.cum_rewards.argmax()
        self.states.update(best_found=copy.deepcopy(self.env_states.observs[ix]))
        self.states.update(best_reward_found=copy.deepcopy(self.states.cum_rewards[ix]))

    def _accumulate_and_update_rewards(self, rewards: np.ndarray):
        """
        Use as reward either the sum of all the rewards received during the \
        current run, or use the last reward value received as reward.

        Args:
            rewards: Array containing the last rewards received by every walker.
        """
        minim_coef = -1 if self.minimize else 1.
        rewards = minim_coef * rewards
        super(MapperWalkers, self)._accumulate_and_update_rewards(rewards=rewards)


class FunctionMapper(Swarm):
    def __init__(self, *args, **kwargs):
        kwargs = update_defaults(
            kwargs, accumulate_rewards=False, walkers=MapperWalkers, model=RandomContinous
        )
        super(FunctionMapper, self).__init__(*args, **kwargs)

    @classmethod
    def from_function(cls, function: Callable, shape: tuple, bounds: list = None, *args, **kwargs):
        env = Function(function=function, bounds=bounds, shape=shape)
        return FunctionMapper(env=lambda: env, *args, **kwargs)

    @property
    def critic(self):
        return self._walkers.critic
