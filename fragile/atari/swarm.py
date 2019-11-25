import numpy as np
from PIL import Image
from plangym import ParallelEnvironment
from plangym.montezuma import Montezuma
from plangym.ray import RayEnv
#from plangym.montezuma import Montezuma
from holoviews.streams import Pipe, Buffer
from streamz.dataframe import DataFrame
from streamz import Stream
import holoviews as hv
import hvplot.pandas
import hvplot.streamz
import numpy as np
import pandas as pd
hv.extension("bokeh")

from fragile.core.env import DiscreteEnv
from fragile.core.dt_sampler import GaussianDt
from fragile.core.models import RandomDiscrete
from fragile.core.states import States
from fragile.core.swarm import Swarm
from fragile.core.walkers import Walkers
from fragile.core.tree import HistoryTree
from fragile.core.utils import resize_frame
from fragile.atari.walkers import MontezumaWalkers
from fragile.atari.critics import MontezumaGrid


class MontezumaSwarm(Swarm):

    def __init__(self, plot_step=10, *args, **kwargs):
        super(MontezumaSwarm, self).__init__(*args, **kwargs)
        self.init_dmap()
        self.plot_step = plot_step
    @property
    def grid(self) -> MontezumaGrid:
        return self.critic

    def init_dmap(self):
        self.image_pipe = Pipe(data=[])
        self.image_dmap = hv.DynamicMap(hv.Image, streams=[self.image_pipe])
        self.image_dmap = self.image_dmap.opts(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
        self.memory_pipe = Pipe(data=[])
        self.memory_dmap = hv.DynamicMap(hv.Image, streams=[self.memory_pipe])
        self.memory_dmap = self.memory_dmap.opts(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
        self.frame_pipe = Pipe(data=[])
        self.frame_dmap = hv.DynamicMap(hv.RGB, streams=[self.frame_pipe])
        self.frame_dmap = self.frame_dmap.opts(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))

    @staticmethod
    def create_swarm(critic_scale:float=1, env_workers: int=8, *args, **kwargs):
        def create_env():
            return Montezuma(
                autoreset=True,
                episodic_live=True,
                min_dt=1,
            )
        env = RayEnv(env_callable=create_env, n_workers=env_workers)
        dt = GaussianDt(min_dt=3, max_dt=1000, loc_dt=6, scale_dt=4)

        swarm = MontezumaSwarm(
            model=lambda x: RandomDiscrete(x, dt_sampler=dt),
            walkers=MontezumaWalkers,
            env=lambda: DiscreteEnv(env),
            tree=None, #HistoryTree,
            critic=MontezumaGrid(scale=critic_scale),
            *args, **kwargs

        )
        return swarm


    @staticmethod
    def plot_grid_over_obs(observation, grid) -> hv.NdOverlay:
        background = observation[50:, :, ].mean(axis=2).astype(bool).astype(int) * 255
        peste = resize_frame(grid.T[::1, ::1], 160, 160, "L")
        peste = peste / peste.max() * 255
        return hv.RGB(background) * hv.Image(peste).opts(alpha=0.7)

    def plot_dmap(self) -> hv.NdOverlay:
        return self.frame_dmap + self.image_dmap * self.memory_dmap.opts(alpha=0.7)

    def stream_dmap(self):
        best_ix = np.random.choice(self.walkers.n)
        best_ix = self.walkers.states.cum_rewards.argmax()
        raw_background = self.walkers.env_states.observs[best_ix, :-3].reshape((210, 160, 3))
        best_room = int(self.walkers.env_states.observs[best_ix, -1])
        grid = self.grid.memory[:, :, best_room]

        background = raw_background[50:, :].mean(axis=2).astype(bool).astype(int) * 255
        grid = resize_frame(grid.T, 160, 160, "L")
        grid = grid / grid.max() * 255
        self.memory_pipe.send(grid)
        self.image_pipe.send(background)
        self.frame_pipe.send(raw_background[50:, :].astype(np.uint8))

    def plot_critic(self) -> hv.NdOverlay:
        ix = np.random.randint(self.walkers.n)
        #best_ix = self.walkers.states.cum_rewards.argmax()
        background = self.walkers.env_states.observs[ix, :-3].reshape((210, 160, 3))
        # best_room = int(self.walkers.env_states.observs[best_ix, -1])

        peste = self.grid.memory[:, :, ix]
        return self.plot_grid_over_obs(background, peste)

    def run_step(self):
        returned = super(MontezumaSwarm, self).run_step()
        if self.walkers.n_iters % self.plot_step == 0:
            self.stream_dmap()
        return returned

