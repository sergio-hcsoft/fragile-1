from typing import Tuple

import holoviews
import numpy
import pandas

from fragile.core.bounds import Bounds
from fragile.core.swarm import Swarm
from fragile.core.utils import relativize
from fragile.dataviz.streaming import Bivariate, Curve, Histogram, Landscape2D

PLOT_NAMES = ()


class BestReward(Curve):
    name = "best_reward"

    def opts(
        self,
        title="Best value found",
        xlabel: str = "iteration",
        ylabel: str = "Best value",
        *args,
        **kwargs
    ):
        super(BestReward, self).opts(title=title, xlabel=xlabel, ylabel=ylabel, *args, **kwargs)

    def get_plot_data(self, swarm: Swarm = None):
        if swarm is None:
            data = pandas.DataFrame({"x": [], "best_val": []}, columns=["x", "best_val"])
        else:
            data = pandas.DataFrame(
                {"x": [int(swarm.walkers.n_iters)], "best_val": [float(swarm.best_reward_found)]},
                columns=["x", "best_val"],
            )
        return data


class SwarmHistogram(Histogram):
    name = "swarm_histogram"

    def __init__(self, margin_high=1.0, margin_low=1.0, *args, **kwargs):
        self.high = margin_high
        self.low = margin_low
        super(SwarmHistogram, self).__init__(*args, **kwargs)

    def opts(self, ylabel: str = "Frequency", *args, **kwargs):
        super(SwarmHistogram, self).opts(ylabel=ylabel, *args, **kwargs)

    def _update_lims(self, X: numpy.ndarray):

        # bounds = Bounds.from_array(X)
        self.xlim = (X.min(), X.max())

    def get_plot_data(self, swarm: Swarm, attr: str):
        data = getattr(swarm.walkers.states, attr) if swarm is not None else numpy.arange(10)
        self._update_lims(data)
        return super(SwarmHistogram, self).get_plot_data(data)


class RewardHistogram(SwarmHistogram):
    name = "reward_histogram"

    def opts(self, title="Reward distribution", xlabel: str = "Reward", *args, **kwargs):
        super(RewardHistogram, self).opts(title=title, xlabel=xlabel, *args, **kwargs)

    def get_plot_data(self, swarm: Swarm):
        return super(RewardHistogram, self).get_plot_data(swarm, "cum_rewards")


class DistanceHistogram(SwarmHistogram):
    name = "distance_histogram"

    def opts(
        self,
        title="Distance distribution",
        xlabel: str = "Distance",
        ylabel: str = "Frequency",
        *args,
        **kwargs
    ):
        super(DistanceHistogram, self).opts(
            title=title, xlabel=xlabel, ylabel=ylabel, *args, **kwargs
        )

    def get_plot_data(self, swarm: Swarm):
        return super(DistanceHistogram, self).get_plot_data(swarm, "distances")


class VirtualRewardHistogram(SwarmHistogram):
    name = "virtual_reward_histogram"

    def opts(
        self,
        title="Virtual reward distribution",
        xlabel: str = "Virtual reward",
        ylabel: str = "Frequency",
        *args,
        **kwargs
    ):
        super(VirtualRewardHistogram, self).opts(
            title=title, xlabel=xlabel, ylabel=ylabel, *args, **kwargs
        )

    def get_plot_data(self, swarm: Swarm):
        return super(VirtualRewardHistogram, self).get_plot_data(swarm, "virtual_rewards")


def has_embedding(swarm: Swarm) -> bool:
    if hasattr(swarm, "critic"):
        if hasattr(swarm.critic, "preprocess_input"):
            return True
    return False


def get_xy_coords(swarm, use_embedding: False) -> numpy.ndarray:

    if use_embedding and has_embedding(swarm):
        X = swarm.critic.preprocess_input(
            env_states=swarm.walkers.env_states,
            walkers_states=swarm.walkers.states,
            model_states=swarm.walkers.model_states,
            batch_size=swarm.walkers.n,
        )
        return X
    elif isinstance(swarm, numpy.ndarray):
        return swarm
    return swarm.walkers.env_states.observs[:, :2]


class SwarmLandscape(Landscape2D):
    name = "swarm_landscape"

    def __init__(
        self, use_embeddings: bool = True, margin_high=1.0, margin_low=1.0, *args, **kwargs
    ):
        self.use_embeddings = use_embeddings
        self.high = margin_high
        self.low = margin_low
        super(SwarmLandscape, self).__init__(*args, **kwargs)

    def get_z_coords(self, swarm: Swarm, X: numpy.ndarray):
        raise NotImplementedError

    def _update_lims(self, swarm, X: numpy.ndarray):
        backup_bounds = swarm.env.bounds if swarm is not None else Bounds.from_array(X)
        bounds = (
            swarm.critic.bounds if has_embedding(swarm) and self.use_embeddings else backup_bounds
        )
        self.xlim, self.ylim = bounds.safe_margin(low=self.low, high=self.high).to_lims()

    def opts(self, xlim="default", ylim="default", *args, **kwargs):
        xlim = self.xlim if xlim == "default" else xlim
        ylim = self.ylim if ylim == "default" else ylim
        return Landscape2D.opts(self, xlim=xlim, ylim=ylim, *args, **kwargs)

    def _get_plot_data_with_defaults(self, swarm: Swarm) -> Tuple:
        if swarm is not None:
            X = get_xy_coords(swarm, self.use_embeddings)
            z = self.get_z_coords(swarm, X)
            self._update_lims(swarm, X)
        else:
            X = numpy.random.standard_normal((10, 2))
            z = numpy.random.standard_normal(10)
        return X, z

    def get_plot_data(self, swarm: Swarm) -> numpy.ndarray:
        X, z = self._get_plot_data_with_defaults(swarm)
        data = X[:, 0], X[:, 1], z
        return super(SwarmLandscape, self).get_plot_data(data)


class RewardLandscape(SwarmLandscape):
    name = "reward_landscape"

    def opts(self, title="Reward landscape", *args, **kwargs):
        return super(RewardLandscape, self).opts(title=title, *args, **kwargs)

    def get_z_coords(self, swarm: Swarm, X: numpy.ndarray = None):
        rewards: numpy.ndarray = relativize(swarm.walkers.states.cum_rewards)
        return rewards


class VirtualRewardLandscape(SwarmLandscape):
    name = "virtual_reward_landscape"

    def opts(self, title="Virtual reward landscape", *args, **kwargs):
        super(VirtualRewardLandscape, self).opts(title=title, *args, **kwargs)

    def get_z_coords(self, swarm: Swarm, X: numpy.ndarray = None):
        virtual_rewards: numpy.ndarray = swarm.walkers.states.virtual_rewards
        return virtual_rewards


class DistanceLandscape(SwarmLandscape):
    name = "distance_landscape"

    def opts(self, title="Distance landscape", *args, **kwargs):
        super(DistanceLandscape, self).opts(title=title, *args, **kwargs)

    def get_z_coords(self, swarm: Swarm, X: numpy.ndarray = None):
        distances: numpy.ndarray = swarm.walkers.states.distances
        return distances


class InvDistanceLandscape(SwarmLandscape):
    name = "inv_distance_landscape"

    def opts(self, title="Inverse distance landscape", *args, **kwargs):
        super(InvDistanceLandscape, self).opts(title=title, *args, **kwargs)

    def get_z_coords(self, swarm: Swarm, X: numpy.ndarray = None):
        distances: numpy.ndarray = 1 / swarm.walkers.states.distances
        return distances


class WalkersDensity(SwarmLandscape):
    name = "walkers_density"

    def __init__(self, use_embeddings: bool = True, *args, **kwargs):
        self.use_embeddings = use_embeddings
        super(WalkersDensity, self).__init__(*args, **kwargs)

    def get_z_coords(self, swarm: Swarm, X: numpy.ndarray):
        pass

    def opts(
        self,
        title="",
        tools="default",
        xlabel: str = "x",
        ylabel: str = "y",
        shared_axes: bool = False,
        framewise: bool = True,
        axiswise: bool = True,
        normalize: bool = True,
        *args,
        **kwargs
    ):
        tools = tools if tools != "default" else ["hover"]
        self.plot = self.plot.opts(
            holoviews.opts.Bivariate(
                tools=tools,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                show_legend=False,
                colorbar=True,
                filled=True,
                *args,
                **kwargs
            ),
            holoviews.opts.Scatter(
                fill_color="red",
                alpha=0.7,
                size=3.5,
                tools=tools,
                xlabel=xlabel,
                ylabel=ylabel,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                *args,
                **kwargs
            ),
            holoviews.opts.NdOverlay(
                normalize=normalize,
                framewise=framewise,
                axiswise=axiswise,
                shared_axes=shared_axes,
            ),
        )

    def get_plot_data(self, swarm: Swarm) -> numpy.ndarray:
        X, z = self._get_plot_data_with_defaults(swarm)
        return X, X[:, 0], X[:, 1], self.xlim, self.ylim

    @staticmethod
    def plot_landscape(data):
        X, x, y, xlim, ylim = data
        mesh = holoviews.Bivariate(X)
        scatter = holoviews.Scatter((x, y))
        contour_mesh = mesh * scatter
        return contour_mesh.redim(
            x=holoviews.Dimension("x", range=xlim), y=holoviews.Dimension("y", range=ylim),
        )


class GridLandscape(SwarmLandscape):
    name = "grid_landscape"

    def opts(self, title="Memory grid values", *args, **kwargs):
        super(GridLandscape, self).opts(title=title, *args, **kwargs)

    @staticmethod
    def plot_landscape(data):
        x, y, xx, yy, z, xlim, ylim = data
        # xx, yy, memory_grid, _, xlim, ylim, x, y = data
        try:
            memory_vals = z.reshape(xx.shape)
        except ValueError:
            memory_vals = numpy.ones_like(xx)
        mesh = holoviews.QuadMesh((xx, yy, memory_vals))
        plot = (mesh * holoviews.Scatter((x, y))).opts(xlim=xlim, ylim=ylim)
        return plot

    def get_z_coords(self, swarm: Swarm, X: numpy.ndarray = None):
        if swarm is None:
            return numpy.ones(self.n_points ** self.n_points)
        if swarm.critic.bounds is None:
            swarm.critic.bounds = Bounds.from_array(X, scale=1.1)
        # target grid to interpolate to
        xi = numpy.linspace(swarm.critic.bounds.low[0], swarm.critic.bounds.high[0], self.n_points)
        yi = numpy.linspace(swarm.critic.bounds.low[1], swarm.critic.bounds.high[1], self.n_points)
        xx, yy = numpy.meshgrid(xi, yi)
        grid = numpy.c_[xx.ravel(), yy.ravel()]
        if swarm.swarm.critic.warmed:
            memory_values = swarm.swarm.critic.model.transform(grid)
            memory_values = numpy.array(
                [
                    swarm.swarm.critic.memory[ix[0], ix[1]].astype(numpy.float32)
                    for ix in memory_values.astype(int)
                ]
            )
        else:
            memory_values = numpy.arange(grid.shape[0])
        return memory_values


class KDELandscape(SwarmLandscape):
    name = "kde_landscape"

    def opts(self, title="Memory grid values", *args, **kwargs):
        super(KDELandscape, self).opts(title=title, *args, **kwargs)

    @staticmethod
    def plot_landscape(data):
        x, y, xx, yy, z, xlim, ylim = data
        # xx, yy, memory_grid, _, xlim, ylim, x, y = data
        try:
            memory_vals = z.reshape(xx.shape)
        except ValueError:
            memory_vals = numpy.ones_like(xx)
        mesh = holoviews.QuadMesh((xx, yy, memory_vals))
        plot = (mesh * holoviews.Scatter((x, y))).opts(xlim=xlim, ylim=ylim)
        return plot

    def get_z_coords(self, swarm: Swarm, X: numpy.ndarray = None):
        if swarm is None:
            return numpy.ones(self.n_points ** self.n_points)
        if swarm.critic.bounds is None:
            swarm.critic.bounds = Bounds.from_array(X, scale=1.1)
        # target grid to interpolate to
        xi = numpy.linspace(swarm.critic.bounds.low[0], swarm.critic.bounds.high[0], self.n_points)
        yi = numpy.linspace(swarm.critic.bounds.low[1], swarm.critic.bounds.high[1], self.n_points)
        xx, yy = numpy.meshgrid(xi, yi)
        grid = numpy.c_[xx.ravel(), yy.ravel()]
        if swarm.swarm.critic.warmed:
            memory_values = swarm.swarm.critic.predict(grid)
            memory_values = relativize(-memory_values)
        else:
            memory_values = numpy.arange(grid.shape[0])
        return memory_values
