from multiprocessing import Pool

# from plangym.montezuma import Montezuma
from holoviews.streams import Pipe, Buffer
from streamz.dataframe import DataFrame
from streamz import Stream
import holoviews as hv
import hvplot.pandas
import hvplot.streamz
import numpy as np
import pandas as pd
import panel as pn
from plangym.montezuma import Montezuma
from plangym.ray import RayEnv

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
from fragile.atari.plots import plot_grid_over_obs, plot_memories, save_memories


pn.extension()
hv.extension("bokeh")


class MontezumaSwarm(Swarm):
    def __init__(
        self,
        plot_step=10,
        dump_every: int = 32,
        max_rooms: int = 23,
        save_rooms: bool = True,
        *args,
        **kwargs
    ):
        super(MontezumaSwarm, self).__init__(*args, **kwargs)
        self.init_dmap()
        self.plot_step = plot_step
        self.displayed_rooms = {}
        self.plot_buffer = []
        self.pool = Pool()
        self.dump_every = dump_every
        self.max_rooms = max_rooms
        self.save_rooms = save_rooms

    @property
    def grid(self) -> MontezumaGrid:
        return self.critic

    @property
    def n_rooms(self):
        return self.grid.memory.shape[-1]

    @property
    def discovered_rooms(self) -> np.ndarray:
        return np.arange(self.n_rooms)[self.grid.memory.sum(axis=(0, 1)) != 0]

    def init_dmap(self):
        self.image_pipe = Pipe(data=[])
        self.image_dmap = hv.DynamicMap(hv.Image, streams=[self.image_pipe])
        self.image_dmap = self.image_dmap.opts(
            xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), xaxis=None, yaxis=None
        )
        self.best_pipe = Pipe(data=[])
        self.best_dmap = hv.DynamicMap(hv.Image, streams=[self.best_pipe])
        self.best_dmap = self.best_dmap.opts(
            xlim=(-0.5, 0.5),
            ylim=(-0.5, 0.5),
            xaxis=None,
            yaxis=None,
            title="Memory",
            colorbar=True,
        )
        self.frame_pipe = Pipe(data=[])
        self.frame_dmap = hv.DynamicMap(hv.RGB, streams=[self.frame_pipe])
        self.frame_dmap = self.frame_dmap.opts(
            xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), xaxis=None, yaxis=None, title="Screen"
        )
        self.table_pipe = Pipe(data=[])
        self.table_dmap = hv.DynamicMap(hv.Table, streams=[self.table_pipe]).opts(
            title="MontezumaRevenge with memory"
        )
        self.memory_pipe = Pipe(data=[])
        self.memory_dmap = hv.DynamicMap(self.plot_memories, streams=[self.memory_pipe]).opts(
            shared_xaxis=False, shared_yaxis=False, normalize=True
        )
        self.top_row = (
            self.table_dmap + self.frame_dmap + self.image_dmap * self.best_dmap.opts(alpha=0.7)
        )

    @staticmethod
    def create_swarm(
        critic_scale: float = 1, env_workers: int = 8, forget_val: float = 0.05, *args, **kwargs
    ) -> "MontezumaSwarm":
        def create_env():
            return Montezuma(autoreset=True, episodic_live=True, min_dt=1)

        env = RayEnv(env_callable=create_env, n_workers=env_workers)
        dt = GaussianDt(min_dt=3, max_dt=1000, loc_dt=6, scale_dt=4)

        swarm = MontezumaSwarm(
            model=lambda x: RandomDiscrete(x, dt_sampler=dt),
            walkers=MontezumaWalkers,
            env=lambda: DiscreteEnv(env),
            critic=MontezumaGrid(scale=critic_scale, forget_val=forget_val),
            *args,
            **kwargs
        )
        return swarm

    @staticmethod
    def plot_memories(data) -> hv.NdOverlay:
        return plot_memories(*data)

    def plot_dmap(self) -> hv.NdOverlay:
        return self.top_row

    def update_displayed_rooms(self):
        for i in self.discovered_rooms:
            rooms = self.walkers.env_states.observs[:, -1]
            room_available = rooms == i
            if room_available.any():
                room_ix = np.argmax(room_available)
                if room_ix not in self.displayed_rooms.keys():
                    frame = self.walkers.env_states.observs[room_ix, :-3].reshape((210, 160, 3))
                    background = frame[50:, :].mean(axis=2).astype(bool).astype(int) * 255
                    self.displayed_rooms[i] = hv.Image(background)

    def stream_dmap(self):
        best_ix = self.walkers.states.cum_rewards.argmax()
        raw_background = self.walkers.env_states.observs[best_ix, :-3].reshape((210, 160, 3))
        best_room = int(self.walkers.env_states.observs[best_ix, -1])
        grid = self.grid.memory[:, :, best_room]

        background = raw_background[50:, :].mean(axis=2).astype(bool).astype(int) * 255
        grid = resize_frame(grid.T, 160, 160, "L")
        grid = grid / grid.max() * 255
        self.best_pipe.send(grid)
        self.image_pipe.send(background)
        self.frame_pipe.send(raw_background[50:, :].astype(np.uint8))
        table = pd.DataFrame(
            {
                "Iteration": self.walkers.n_iters,
                "Max reward sampled": self.walkers.states.best_reward_found,
                "Discovered rooms": len(self.discovered_rooms),
            },
            index=[0],
        )
        self.table_pipe.send(table)
        self.update_displayed_rooms()
        if self.save_rooms:
            data = self.displayed_rooms.copy(), self.grid.memory.copy(), int(self.walkers.n_iters)
            self.plot_buffer.append(data)
            if self.walkers.n_iters % self.dump_every == 0:
                self.pool.map(save_memories, list(self.plot_buffer))
                del self.plot_buffer
                self.plot_buffer = []

    def run_step(self):
        returned = super(MontezumaSwarm, self).run_step()
        if self.walkers.n_iters % self.plot_step == 0:
            self.stream_dmap()
        return returned
