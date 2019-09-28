import numpy as np
from sklearn.neighbors import KernelDensity
from fragile.core.base_classes import BaseCritic, States
from fragile.core.utils import relativize


class GaussianRepulssion(BaseCritic):

    def __init__(self, warmup:int=100, refresh_rate: int=1000, *args, **kwargs):
        self.refresh_rate = refresh_rate
        self._model_args = args
        self._model_kwargs = kwargs
        self.warmup = warmup
        self._warmed = False
        self.kde = KernelDensity(*args, **kwargs)
        self.buffer = []
        self._epoch = 0

    def calculate(
        self,
        env_states: States,
        walkers_states: "StatesWalkers",
        batch_size: int = None,
        model_states: States = None,

    ) -> np.ndarray:
        self._epoch += 1
        self.buffer.append(env_states.observs)
        if self._epoch < self.warmup and not self._warmed:
            score = np.ones(env_states.n)
            walkers_states.update(critic_score=score)
            return
        elif (self._epoch == self.refresh_rate and self._warmed or
              self._epoch == self.warmup and not self._warmed):
            self.kde = self.kde.fit(np.concatenate(self.buffer, axis=0))
            self._warmed = True
            self._epoch = 0
            self.buffer = []
        probs = self.kde.score_samples(env_states.observs)
        score = relativize(-probs)
        walkers_states.update(critic_score=score)

    def reset(self, batch_size: int = 1, model_states: States = None, *args, **kwargs) -> States:
        self.buffer = []
        self._epoch = 0
        self.kde = KernelDensity(*self._model_args, **self._model_kwargs)
        self._warmed = False

    def update(self, *args, **kwargs):
        pass

