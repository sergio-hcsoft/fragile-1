from collections import deque
import copy

from fragile.core.utils import random_state
from fragile.distributed.export_swarm import ExportedWalkers


class BestWalker(ExportedWalkers):
    def __init__(self, minimize: bool = False):
        super(BestWalker, self).__init__(1)
        self.minimize = minimize

    def update_best(self, walkers: ExportedWalkers):
        curr_best = self.get_best_reward(self.minimize)
        other_best = walkers.get_best_reward(self.minimize)
        other_improves = curr_best > other_best if self.minimize else curr_best < other_best
        if other_improves:
            ix = walkers.get_best_index(self.minimize)
            self.states = copy.deepcopy(walkers.states[ix])
            self.observs = copy.deepcopy(walkers.observs[ix])
            self.rewards = copy.deepcopy(walkers.rewards[ix])
            self.id_walkers = copy.deepcopy(walkers.id_walkers[ix])


class ParamServer:
    def __init__(self, max_len: int = 20, minimize: bool = False, add_global_best: bool = True):
        self._max_len = max_len
        self.add_global_best = add_global_best
        self.minimize = minimize
        self.buffer = deque([], self._max_len)
        self.best = BestWalker(minimize=self.minimize)

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def max_len(self) -> int:
        return self._max_len

    def reset(self):
        self.buffer = deque([], self._max_len)
        self.best = BestWalker(minimize=self.minimize)

    def exchange_walkers(self, walkers: ExportedWalkers) -> ExportedWalkers:
        self.import_walkers(walkers)
        if len(self) == 0:
            return ExportedWalkers(0)
        return self.export_walkers()

    def import_walkers(self, walkers: ExportedWalkers):
        self._track_best_walker(walkers)
        self.buffer.append(walkers)

    def export_walkers(self) -> ExportedWalkers:
        index = random_state.randint(0, len(self))
        walkers = self.buffer[index]
        if self.add_global_best:
            walkers = self._add_best_to_exported(walkers)
        return walkers

    def _track_best_walker(self, walkers: ExportedWalkers):
        self.best.update_best(walkers=walkers)

    def _add_best_to_exported(self, walkers: ExportedWalkers) -> ExportedWalkers:
        index = random_state.randint(0, len(walkers))
        walkers.rewards[index] = self.best.rewards
        walkers.id_walkers[index] = self.best.id_walkers
        walkers.observs[index] = self.best.observs
        walkers.states[index] = self.best.states
        return walkers
