from typing import Callable, Optional

import numpy as np

from fragile.core.models import Bounds, RandomContinous
from fragile.core.swarm import Swarm
from fragile.core.walkers import Walkers, States, StatesWalkers
from fragile.optimize.env import Function, Minimizer


try:
    from IPython.core.display import clear_output
except ImportError:

    def clear_output(**kwargs):
        """If not using jupyter notebook do nothing."""
        pass


class FunctionMapper(Swarm):
    def __init__(
        self,
        walkers=Walkers,
        model=RandomContinous,
        accumulate_rewards: bool = False,
        minimize: bool = True,
        start_same_pos: bool = False,
        *args,
        **kwargs,
    ):
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
        cls, function: Callable, shape: tuple, bounds: Bounds = None, *args, **kwargs
    ) -> "FunctionMapper":
        env = Function(function=function, bounds=bounds, shape=shape)
        return FunctionMapper(env=lambda: env, *args, **kwargs)

    def __repr__(self):
        return "{}\n{}".format(self.env.__repr__(), super(FunctionMapper, self).__repr__())

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
        self.reset(model_states=model_states, env_states=env_states)
        self.epoch = 0
        while not self.calculate_end_condition():
            try:
                self.run_step()
                if self.epoch % print_every == 0:
                    print(self)
                    clear_output(True)
                self.epoch += 1
            except KeyboardInterrupt as e:
                break

    def reset(
        self,
        walkers_states: StatesWalkers = None,
        model_states: States = None,
        env_states: States = None,
    ):
        super(FunctionMapper, self).reset(
            walkers_states=walkers_states, model_states=model_states, env_states=env_states
        )

        if self.start_same_pos:
            self.walkers.env_states.observs[:] = self.walkers.env_states.observs[0]
            self.walkers.env_states.states[:] = self.walkers.env_states.states[0]


class LennardMapper(FunctionMapper):
    def __init__(
        self,
        best_walker: tuple = None,
        best_reward: float = np.inf,
        best_obs: Optional[np.ndarray] = None,
        *args,
        **kwargs
    ):
        best_state, best_obs, best_reward = (
            best_walker if best_walker is not None else (-10, best_obs, best_reward)
        )

        super(LennardMapper, self).__init__(
            true_best=best_reward, true_best_reward=best_obs, true_best_end=False, *args, **kwargs
        )
        self.minimizer = Minimizer(function=self.env)

    def run_step(self):
        self.step_and_update_best()
        self._get_real_best()
        self.balance_and_prune()

    def _get_real_best(self):
        best = self.walkers.env_states.observs[-1]
        best, reward = self.minimizer.minimize_point(best)
        self.walkers.states.update(
            true_best_reward=reward,
            true_best=best,
            true_best_end=self.walkers.states.end_condition[-1],
        )
