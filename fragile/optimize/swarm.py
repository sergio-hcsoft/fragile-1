from typing import Callable

from fragile.core.models import Bounds, RandomContinous
from fragile.core.swarm import Swarm
from fragile.core.walkers import Walkers
from fragile.optimize.env import Function


class FunctionMapper(Swarm):

    def __init__(self, walkers=Walkers, model=RandomContinous,
                 accumulate_rewards: bool = False, minimize: bool = True, *args, **kwargs):
        super(FunctionMapper, self).__init__(walkers=walkers, model=model,
                                             accumulate_rewards=accumulate_rewards,
                                             minimize=minimize,
                                             *args, **kwargs)

    @classmethod
    def from_function(cls, function: Callable, shape: tuple, bounds: Bounds = None,
                      *args, **kwargs) -> "FunctionMapper":
        env = Function(function=function, bounds=bounds, shape=shape)
        return FunctionMapper(env=lambda: env, *args, **kwargs)

    def __repr__(self):
        return "{}\n{}".format(self.env.__repr__(), super(FunctionMapper, self).__repr__())



