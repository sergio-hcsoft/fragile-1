import copy
from typing import Callable

import numpy as np

from fragile.core.models import RandomContinous
from fragile.core.states import BaseStates
from fragile.core.swarm import Swarm
from fragile.core.utils import update_defaults
from fragile.core.walkers import float_type, Walkers
from fragile.optimize.encoder import Encoder
from fragile.optimize.env import Function


class MapperWalkers(Walkers):
    def __init__(self, encoder: Encoder, *args, **kwargs):
        """
        Initialize a :class:`MapperWalkers`.

        Args:
            encoder: Encoder that will be used to calculate the pests.
            *args:
            **kwargs:
        """
        super(MapperWalkers, self).__init__(*args, **kwargs)
        self.encoder = encoder
        pests = np.zeros(self.n, dtype=float_type)
        # Add data specific to the child class in the StatesWalkers class as new attributes.
        self.states.update(pests=pests, best_reward_found=-1e10, best_found=None)

    def __repr__(self):
        text = "Best reward found: {:.5f} at position: {}," "Encoder: \n {}".format(
            float(self.states.best_reward_found), self.states.best_found, self.encoder
        )
        return text + super(MapperWalkers, self).__repr__()

    def calculate_virtual_reward(self):
        super(MapperWalkers, self).calculate_virtual_reward()
        pests = self.encoder.calculate_pest(self)
        virt_rew = self.states.virtual_rewards * self.states.pests
        self.states.update(virtual_rewards=virt_rew, pests=pests)

    def balance(self):
        self.update_best()
        returned = super(MapperWalkers, self).balance()
        self.encoder.update(
            walkers_states=self.states, model_states=self.model_states, env_states=self.env_states
        )
        return returned

    def update_best(self):
        ix = self.states.cum_rewards.argmax()
        best = self.observs[ix].copy()
        best_reward = float(self.cum_rewards[ix])
        best_is_alive = not bool(self.states.ends[ix])
        if self.best_reward_found < best_reward and best_is_alive:
            self.states.update(best_reward_found=best_reward)
            self.states.update(best_found=best)

    def reset(self, env_states: BaseStates = None, model_states: BaseStates = None):
        super(MapperWalkers, self).reset(env_states=env_states, model_states=model_states)
        ix = self.cum_rewards.argmax()
        self.states.update(best_found=copy.deepcopy(self.observs[ix]))
        self.states.update(best_reward_found=copy.deepcopy(self.states.cum_rewards[ix]))


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
    def encoder(self):
        return self._walkers.encoder

    def record_visited(self):
        observs = self.walkers.observs
        x, y = observs[:, 0].tolist(), observs[:, 1].tolist()
        rewards = self.walkers.rewards.flatten().tolist()
        self.visited_x.extend(x[:-1])
        self.visited_y.extend(y[:-1])
        self.visited_rewards.extend(rewards[:-1])
