from typing import Callable

import torch

from fragile.core.models import RandomContinous
from fragile.core.states import BaseStates
from fragile.core.swarm import Swarm
from fragile.core.walkers import Walkers
from fragile.optimize.encoder import Encoder
from fragile.optimize.env import Function


class MapperWalkers(Walkers):
    def __init__(self, n_vectors, timout: int = 1000, *args, **kwargs):
        self.encoder = Encoder(n_vectors=n_vectors, timeout=timout)
        super(MapperWalkers, self).__init__(*args, **kwargs)
        self.pwise_distance = self._distances
        self._pwise_dist_module = torch.nn.PairwiseDistance().to(self.device)
        self.best_found = None
        self.best_reward_found = -1e20
        self._n_cloned_vectors = 0
        self._score_vectors = 0

    def __repr__(self):
        text = (
            "Best reward found: {:.5f} at position: {}, and {} "
            "cloned_vectors \n Encoder: \n {}".format(
                float(self.best_reward_found),
                self.best_found,
                self._n_cloned_vectors,
                self.encoder,
            )
        )
        return text + super(MapperWalkers, self).__repr__()

    def get_clone_vectors(self):
        vectors = []
        starts = self.observs[self.will_clone]
        ends = self.observs[self.compas_ix][self.will_clone]
        clones = int(self.will_clone.sum())
        for i in range(clones):
            si, ei = starts[i].detach().clone(), ends[i].detach().clone()
            vectors.append((si, ei))
        self._n_cloned_vectors = len(vectors)
        return vectors

    def balance(self):
        self.update_best()
        returned = super(MapperWalkers, self).balance()
        new_vectors = self.get_clone_vectors()
        self.encoder.update_bases(new_vectors)
        self.encoder.remove_bases(self.observs)

        return returned

    def update_best(self):
        ix = self.cum_rewards.argmax()
        best = self.observs[ix].detach().cpu().clone()
        best_reward = self.cum_rewards[ix]
        if self.best_reward_found < best_reward:
            self.best_reward_found = float(best_reward.cpu())
            self.best_found = best

    def _distances(self, state_1, state_2):
        with torch.no_grad():
            if len(self.encoder) < self.encoder.n_vectors:
                return self._pwise_dist_module(state_1, state_2)

            x = self.encoder.encode(state_1)
            y = self.encoder.encode(state_2)
            return torch.sum((y - x) ** 2, 1).reshape(-1, 1)

    def update_clone_probs(self):
        super(MapperWalkers, self).update_clone_probs()
        self.will_clone[-1] = 0

    def reset(self, env_states: BaseStates = None, model_states: BaseStates = None):
        super(MapperWalkers, self).reset(env_states=env_states, model_states=model_states)
        ix = self.cum_rewards.argmax()
        self.best_found = self.observs[ix].detach().cpu().clone()
        self.best_reward_found = float(-1e20)


class FunctionMapper(Swarm):
    def __init__(
        self,
        function: Callable,
        shape: tuple,
        model: Callable = RandomContinous,
        bounds: list = None,
        *args,
        **kwargs
    ):
        env = Function(function=function, bounds=bounds, shape=shape)
        super(FunctionMapper, self).__init__(
            env=lambda: env, model=model, walkers=MapperWalkers, *args, **kwargs
        )

    def __repr__(self):

        return super(Swarm, self).__repr__()
