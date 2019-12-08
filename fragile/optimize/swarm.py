from typing import Callable, Optional

import numpy as np

from fragile.core.models import Bounds, RandomContinous
from fragile.core.swarm import Swarm
from fragile.core.walkers import Walkers
from fragile.optimize.env import Function, Minimizer


class FunctionMapper(Swarm):
    def __init__(
        self,
        walkers=Walkers,
        model=RandomContinous,
        accumulate_rewards: bool = False,
        minimize: bool = True,
        *args,
        **kwargs
    ):
        super(FunctionMapper, self).__init__(
            walkers=walkers,
            model=model,
            accumulate_rewards=accumulate_rewards,
            minimize=minimize,
            *args,
            **kwargs
        )

    @classmethod
    def from_function(
        cls, function: Callable, shape: tuple, bounds: Bounds = None, *args, **kwargs
    ) -> "FunctionMapper":
        env = Function(function=function, bounds=bounds, shape=shape)
        return FunctionMapper(env=lambda: env, *args, **kwargs)

    def __repr__(self):
        return "{}\n{}".format(self.env.__repr__(), super(FunctionMapper, self).__repr__())


class LennardMapper(FunctionMapper):
    def __init__(
        self,
        best_walker: tuple = None,
        best_reward: float = -1e10,
        best_obs: Optional[np.ndarray] = None,
        *args,
        **kwargs
    ):
        best_state, best_obs, best_reward = (
            best_walker if best_walker is not None else (-1e10, None, None)
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
