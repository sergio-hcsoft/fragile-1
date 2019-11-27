from multiprocessing import Pool
from typing import Callable

import numpy as np
import ray

from fragile.core.states import States
from fragile.optimize.env import Function as SequentialFunction

def split_similar_chunks(vector: list, n_chunks: int):
    chunk_size = int(np.ceil(len(vector) / n_chunks))
    for i in range(0, len(vector), chunk_size):
        yield vector[i : i + chunk_size]

@ray.remote
class RemoteFunction:

    def __init__(self, env_callable: Callable):
        self.function = env_callable().function

    def function(self, points: np.ndarray):
        return self.function(points)


class Function(SequentialFunction):

    def __init__(self, env_callable: Callable, n_workers: int = 1):
        self.n_workers = n_workers
        self.workers = [RemoteFunction.remote(env_callable) for _ in range(n_workers)]
        self.local_function = env_callable()
        self.pool = Pool(n_workers)

    def __getattr__(self, item):
        return getattr(self.local_function, item)

    def step(self, model_states: States, env_states: States) -> States:
        """
        Sets the environment to the target states by applying the specified actions an arbitrary
        number of time steps.

        Args:
            model_states: States corresponding to the model data.
            env_states: States class containing the state data to be set on the Environment.

        Returns:
            States containing the information that describes the new state of the Environment.
        """
        new_points = (
            # model_states.actions * model_states.dt.reshape(env_states.n, -1) + env_states.observs
            model_states.actions + env_states.observs
        )
        ends = self.calculate_end(points=new_points)
        rewards = self.parallel_function(new_points)

        last_states = self._get_new_states(new_points, rewards, ends, model_states.n)
        return last_states

    def parallel_function(self, points):
        #reward_ids = [env.function.remote(p) for env, p in
        #              zip(self.workers, split_similar_chunks(points, self.n_workers))]
        #rewards = ray.get(reward_ids)
        rewards = self.pool.map(self.local_function.function,
                                split_similar_chunks(points, self.n_workers))
        return np.concatenate([r.flatten() for r in rewards])
