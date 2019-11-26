import numpy as np
import pandas as pd
from plangym import AtariEnvironment, ParallelEnvironment

from fragile.ray.swarm import DistributedSwarm


class DistributedMontezuma(DistributedSwarm):

    def stream_progress(self, state, observation, reward):
        example = pd.DataFrame({"reward": [reward]}, index=[self.n_iters // self.n_swarms])
        self.stream.emit(example)
        obs = observation[:-3].reshape((210, 160, 3)).astype(np.uint8)
        self.frame_pipe.send(obs)


class DistributedRam(DistributedSwarm):

    def __init__(self, swarm, *args, **kwargs):
        super(DistributedRam, self).__init__(swarm=swarm, *args, **kwargs)
        self.local_swarm = swarm()
        env = self.local_swarm.env
        env_name = env.name if isinstance(env, ParallelEnvironment) else env._env.name
        self.local_env = AtariEnvironment(name=env_name, clone_seeds=True)
        self.local_env.reset()

    def image_from_state(self, state):
        self.local_env.set_state(state.astype(np.uint8).copy())
        self.local_env.step(0)
        return np.asarray(self.local_env._env.ale.getScreenRGB(), dtype=np.uint8)

    def stream_progress(self, state, observation, reward):
        example = pd.DataFrame({"reward": [reward]}, index=[self.n_iters // self.n_swarms])
        self.stream.emit(example)
        obs = self.image_from_state(state)
        self.frame_pipe.send(obs)