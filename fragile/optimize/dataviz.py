import holoviews as hv
from holoviews.streams import Pipe, Buffer
import numpy as np
import pandas as pd
from umap import UMAP

from fragile.optimize.swarm import FunctionMapper


class PlotSwarm(FunctionMapper):
    def __init__(self, *args, **kwargs):
        super(PlotSwarm, self).__init__(*args, **kwargs)
        self.pipe_walkers = Pipe(
            data=pd.DataFrame(columns=["x", "y", "reward", "virtual_reward", "dead"])
        )
        example = pd.DataFrame({"x": [], "best_val": []}, columns=["x", "best_val"])
        self.buffer_best = Buffer(example, length=10000, index=False)
        example = pd.DataFrame(
            {"x": [], "y": [], "count": [], "value": []}, columns=["x", "y", "count", "value"]
        )
        self.buffer_walker = Buffer(example, length=10000, index=False)
        self.buffer_path_best = Buffer(example.copy(), length=10000, index=False)
        self.umap = UMAP(n_components=2)

    def run_step(self):
        self.walkers.fix_best()

        self.step_walkers()
        old_ids, new_ids = self.walkers.balance()
        self.prune_tree(old_ids=set(old_ids.tolist()), new_ids=set(new_ids.tolist()))
        self.stream_walkers()
        self.stream_best()

    def stream_best(self):
        best_val = pd.DataFrame(
            {"x": [int(self.walkers.n_iters)], "best_val": [self.best_reward_found]},
            columns=["x", "best_val"],
        )

        self.buffer_best.send(best_val)
        # emb = self.umap.transform(self.best_found.reshape(1, -1))
        # x, y = emb[0, 0], [emb[0, 1]]
        x, y = self.best_found[0], self.best_found[1]
        best_path = pd.DataFrame(
            {
                "x": [x],
                "y": [y],
                "count": [self.walkers.n_iters],
                "value": [self.best_reward_found],
            },
            columns=["x", "y", "count", "value"],
        )
        self.buffer_path_best.send(best_path)

    def stream_walkers(self):
        # self.umap = self.umap.fit(self.walkers.env_states.observs)
        # emb = self.umap.transform(self.walkers.env_states.observs)
        df = pd.DataFrame(columns=["x", "y", "reward", "virtual_reward", "dead"])
        df["x"] = self.walkers.env_states.observs[:, 0]
        df["y"] = self.walkers.env_states.observs[:, 1]
        df["virtual_reward"] = self.walkers.states.virtual_rewards
        df["virtual_reward"] = df[["virtual_reward"]].apply(lambda x: 2.5 * (1 + x / x.max()))
        df["reward"] = self.walkers.states.cum_rewards
        df["dead"] = self.walkers.states.end_condition.astype(int)
        self.pipe_walkers.send(df)

        best_path = pd.DataFrame(
            {
                "x": [df.loc[0, "x"]],
                "y": [df.loc[0, "y"]],
                "count": [self.walkers.n_iters],
                "value": [float(df.loc[0, "reward"])],
            },
            columns=["x", "y", "count", "value"],
        )
        self.buffer_walker.send(best_path)

    def plot_best_evolution(self):
        curve_dmap = hv.DynamicMap(hv.Curve, streams=[self.buffer_best])
        curve_dmap = curve_dmap.opts(
            tools=["hover"], title="Best value found", xlabel="iteration", ylabel="Best value"
        )
        return curve_dmap

    def plot_best_path(self) -> hv.Scatter:
        best_curve_dmap = hv.DynamicMap(hv.Curve, streams=[self.buffer_path_best])
        best_curve_dmap = best_curve_dmap.opts(
            hv.opts.Curve(
                line_width=1,
                color="green",
                xlim=(self.env.bounds.low[0], self.env.bounds.high[0]),
                title="Evolution of best solution",
                ylim=(self.env.bounds.low[1], self.env.bounds.high[1]),
            )
        )
        best_point_dmap = hv.DynamicMap(hv.Points, streams=[self.buffer_path_best]).opts()

        best_path = (best_curve_dmap * best_point_dmap).opts(
            hv.opts.Points(color="count", line_color="black", size=6, padding=0.1)
        )
        return best_path

    def plot_walker_path(self):
        walker_curve_dmap = hv.DynamicMap(hv.Curve, streams=[self.buffer_walker])
        walker_curve_dmap = walker_curve_dmap.opts(line_width=1, color="black")
        walker_point_dmap = hv.DynamicMap(hv.Points, streams=[self.buffer_walker])
        walker_point_dmap = walker_point_dmap.opts(
            color="value",
            line_color="black",
            size=5,
            xlim=(self.env.bounds.low[0], self.env.bounds.high[0]),
            ylim=(self.env.bounds.low[1], self.env.bounds.high[1]),
            padding=0.1,
            title="Evolution of the first walker",
        )

        walker_path = walker_curve_dmap * walker_point_dmap
        return walker_path

    def plot_walkers(self) -> hv.Scatter:
        walkers_plot = hv.DynamicMap(hv.Scatter, streams=[self.pipe_walkers])
        walkers_plot = walkers_plot.opts(
            tools=["hover"],
            title="Walkers",
            xlabel="first dimension of data",
            ylabel="second dimension of data",
            xlim=(self.env.bounds.low[0], self.env.bounds.high[0]),
            ylim=(self.env.bounds.low[1], self.env.bounds.high[1]),
            color="reward",
            size="virtual_reward",  # marker="dead",
            cmap="viridis",
        )
        return walkers_plot
