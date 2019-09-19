import copy
from typing import Callable

import numpy as np

from fragile.core.models import Bounds, RandomContinous
from fragile.core.states import States
from fragile.core.swarm import Swarm
from fragile.core.utils import relativize, update_defaults
from fragile.core.walkers import float_type, StatesWalkers, Walkers
from fragile.optimize.encoder import Critic
from fragile.optimize.env import Function


class MapperWalkers(Walkers):
    def __init__(self, encoder: Critic = None, minimize: bool = True, *args, **kwargs):
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
        rewards = -1 * self.states.cum_rewards if self.minimize else self.states.cum_rewards
        processed_rewards = relativize(rewards)
        virt_rw = processed_rewards ** self.reward_scale * self.states.distances ** self.dist_scale
        self.update_states(virtual_rewards=virt_rw, processed_rewards=processed_rewards)
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
        rewards = self.states.cum_rewards
        ix = rewards.argmin() if self.minimize else rewards.argmax()
        best = self.env_states.observs[ix].copy()
        best_reward = float(self.states.cum_rewards[ix])
        best_is_alive = not bool(self.env_states.ends[ix])
        has_improved = (self.states.best_reward_found > best_reward if self.minimize else
                        self.states.best_reward_found < best_reward)
        if has_improved and best_is_alive:
            self.states.update(best_reward_found=best_reward)
            self.states.update(best_found=best)

    def fix_best(self):
        self.env_states.observs[-1] = self.states.best_found
        self.env_states.rewards[-1] = self.states.best_reward_found

    def reset(self, env_states: States = None, model_states: States = None):
        super(MapperWalkers, self).reset(env_states=env_states, model_states=model_states)
        rewards = self.env_states.rewards
        ix = rewards.argmin() if self.minimize else rewards.argmax()
        self.states.update(best_found=copy.deepcopy(self.env_states.observs[ix]))
        self.states.update(best_reward_found=np.inf if self.minimize else -np.inf)

    def _accumulate_and_update_rewards(self, rewards: np.ndarray):
        """
        Use as reward either the sum of all the rewards received during the \
        current run, or use the last reward value received as reward.

        Args:
            rewards: Array containing the last rewards received by every walker.
        """
        super(MapperWalkers, self)._accumulate_and_update_rewards(rewards=rewards)


class FunctionMapper(Swarm):

    def __init__(self, *args, **kwargs):
        kwargs = update_defaults(
            kwargs, accumulate_rewards=False, walkers=MapperWalkers, model=RandomContinous
        )
        super(FunctionMapper, self).__init__(*args, **kwargs)

    @property
    def walkers(self) -> MapperWalkers:
        return super(FunctionMapper, self).walkers

    @property
    def best_found(self):
        return self.walkers.states.best_found

    @property
    def best_reward_found(self):
        return self.walkers.states.best_reward_found

    @classmethod
    def from_function(cls, function: Callable, shape: tuple, bounds: Bounds = None,
                      *args, **kwargs) -> "FunctionMapper":
        env = Function(function=function, bounds=bounds, shape=shape)
        return FunctionMapper(env=lambda: env, *args, **kwargs)

    @property
    def critic(self):
        return self._walkers.critic

    def run_swarm(
        self,
        model_states: States = None,
        env_states: States = None,
        walkers_states: StatesWalkers = None,
        print_every: int = 1e100,
    ):
        """
        Run a new search process.

        Args:
            model_states: States that define the initial state of the environment.
            env_states: States that define the initial state of the model.
            walkers_states: States that define the internal states of the walkers.
            print_every: Display the algorithm progress every `print_every` epochs.
        Returns:
            None.

        """
        from IPython.core.display import clear_output
        self.reset(model_states=model_states, env_states=env_states)
        self.epoch = 0
        while not self.walkers.calculate_end_condition():
            try:
                self.walkers.fix_best()
                self.step_walkers()
                old_ids, new_ids = self.walkers.balance()
                self.prune_tree(old_ids=set(old_ids.tolist()), new_ids=set(new_ids.tolist()))
                if self.epoch % print_every == 0:
                    print(self.walkers)
                    clear_output(True)
                self.epoch += 1
            except KeyboardInterrupt as e:
                break
