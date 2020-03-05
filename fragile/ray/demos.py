import holoviews as hv
from holoviews.streams import Pipe
import hvplot.pandas
import hvplot.streamz
import numpy as np
import pandas as pd
from plangym import AtariEnvironment, ParallelEnvironment
from streamz import Stream
from streamz.dataframe import DataFrame

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


class DistributedOptimizer(DistributedSwarm):
    def stream_progress(self, state, observation, reward):
        example = pd.DataFrame({"reward": [reward]}, index=[self.n_iters // self.n_swarms])
        self.stream.emit(example)
        msg_obs = "Best solution found:\n {}".format(np.round(observation, 2).tolist())
        msg_reward = "Best value found: {:.4f}".format(reward)
        data = [[0, 1, msg_reward], [0, 2, msg_obs]]
        self.frame_pipe.send(pd.DataFrame(data, columns=["x", "y", "label"]))

    def init_plot(self):
        self.frame_pipe = Pipe(data=[])
        self.frame_dmap = hv.DynamicMap(hv.Labels, streams=[self.frame_pipe])
        self.frame_dmap = self.frame_dmap.opts(
            xlim=(-10, 10),
            ylim=(0.5, 2.5),
            height=200,
            width=500,
            xaxis=None,
            yaxis=None,
            title="Best solution",
        )
        example = pd.DataFrame({"reward": []})
        self.stream = Stream()
        self.buffer_df = DataFrame(stream=self.stream, example=example)
        self.score_dmap = self.buffer_df.hvplot(y=["reward"]).opts(
            height=200, width=400, title="Best value found"
        )


class DistributedLennardJonnes(DistributedSwarm):
    def stream_progress(self, state, observation, reward):
        ix = self.n_iters // self.n_swarms
        example = pd.DataFrame({"reward": [reward]}, index=[ix])
        self.stream.emit(example)
        # msg_obs = "Best solution found:\n {}".format(numpy.round(observation, 2).tolist())
        msg_reward = "Best value found: {:.4f}".format(reward)
        data = [[ix * 0.5, self.init_reward - 3, msg_reward]]
        self.label_pipe.send(pd.DataFrame(data, columns=["x", "y", "label"]))
        if self.best[0] is not None:
            x = self.best[0].reshape(-1, 3)
            d = {
                "x": x[:, 0].copy().tolist(),
                "y": x[:, 1].copy().tolist(),
                "z": x[:, 2].copy().tolist(),
            }
            self.best_pipe.send(d)

    def plot(self):
        return self.best_dmap + self.label_dmap * self.score_dmap

    def init_plot(self):
        hv.extension("plotly")
        self.best_pipe = Pipe(data=[])
        self.best_dmap = hv.DynamicMap(hv.Scatter3D, streams=[self.best_pipe])
        self.best_dmap = self.best_dmap.opts(
            xlim=(-2, 2),
            ylim=(-2, 4),
            color="red",
            alpha=0.7,
            # height=600, width=600,
            xaxis=None,
            yaxis=None,
            title="Best solution",
        )
        self.label_pipe = Pipe(data=[])
        self.label_dmap = hv.DynamicMap(hv.Labels, streams=[self.label_pipe])
        self.label_dmap = self.label_dmap.opts(
            # height=200, width=400,
            xaxis=None,
            yaxis=None,
            title="Best solution",
        )
        example = pd.DataFrame({"reward": []})
        self.stream = Stream()
        self.buffer_df = DataFrame(stream=self.stream, example=example)
        self.score_dmap = self.buffer_df.hvplot(y=["reward"]).opts(  # height=200, width=400,
            title="Best value found"
        )
