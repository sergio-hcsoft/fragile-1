from scipy.interpolate import griddata
import holoviews as hv
from holoviews.streams import Pipe, Buffer
import numpy as np
import pandas as pd

from fragile.core.bounds import Bounds
from fragile.core.states import States
from fragile.core.swarm import Swarm
from fragile.core.walkers import StatesWalkers
from fragile.core.utils import relativize


try:
    from IPython.core.display import clear_output
except ImportError:

    def clear_output(**kwargs):
        """If not using jupyter notebook do nothing."""
        pass


class BasePlotSwarm:

    PLOT_TYPES = {}
    scatter_opts = dict(
        fill_color="red",
        alpha=0.7,
        size=3.5,
        shared_axes=False,
        colorbar=False,
        normalize=True,
        framewise=True,
        axiswise=True,
        xlim=(None, None),
        ylim=(None, None),
    )

    def __init__(
        self, swarm: Swarm, display_plots="all", stream_interval: int = 100,
    ):
        self.swarm: Swarm = swarm
        self.display_plots = display_plots
        self.plots = {}
        self._init_plots()
        self.stream_interval = stream_interval

    def __getattr__(self, item):
        return getattr(self.swarm, item)

    def __repr__(self):
        return self.swarm.__repr__()

    def run_swarm(
        self,
        model_states: States = None,
        env_states: States = None,
        walkers_states: StatesWalkers = None,
        print_every: int = 1e100,
    ):
        """
        Run a new search process.

        Args:
            model_states: States that define the initial state of the environment.
            env_states: States that define the initial state of the model.
            walkers_states: States that define the internal states of the walkers.
            print_every: Display the algorithm progress every `print_every` epochs.
        Returns:
            None.

        """
        self.swarm.reset(model_states=model_states, env_states=env_states)
        self.swarm.epoch = 0
        while not self.calculate_end_condition():
            try:
                self.run_step()
                if self.swarm.epoch % print_every == 0:
                    print(self)
                    clear_output(True)
                self.swarm.epoch += 1
            except KeyboardInterrupt as e:
                break

    def run_step(self):
        self.swarm.run_step()
        if self.walkers.n_iters % self.stream_interval == 0:
            self.stream_plots()

    def _init_plots(self):
        plot_types = self.PLOT_TYPES if self.display_plots == "all" else self.display_plots
        self.display_plots = plot_types
        self.plots = {}
        for pt in self.display_plots:
            if pt in self.PLOT_TYPES:
                plot = getattr(self, "_init_%s_plot" % pt)()
                setattr(self, "%s_dmap" % pt, plot)
                self.plots[pt] = plot

    def stream_plots(self):
        for pt in self.display_plots:
            getattr(self, "_stream_%s_data" % pt)()

    def plot_dmap(self, ignore: list = None):
        ignore = ignore if ignore is not None else []
        plots = [getattr(self, "%s_dmap" % k) for k in self.display_plots if k not in ignore]
        plot = plots[0]
        for p in plots[1:]:
            plot = plot + p
        return plot


class Plot1DSwarm(BasePlotSwarm):
    PLOT_TYPES = {"best", "reward_hist", "distance_hist", "vr_hist"}

    def __init__(self, swarm, bins: int = 20, *args, **kwargs):
        self.bins = bins
        super(Plot1DSwarm, self).__init__(swarm=swarm, *args, **kwargs)

    def _init_best_plot(self):
        example = pd.DataFrame({"x": [], "best_val": []}, columns=["x", "best_val"])
        self.best_stream = Buffer(example, length=10000, index=False)
        curve_dmap = hv.DynamicMap(hv.Curve, streams=[self.best_stream])
        curve_dmap = curve_dmap.opts(
            hv.opts.Curve(
                tools=["hover"],
                title="Best value found",
                xlabel="iteration",
                ylabel="Best value",
                shared_axes=False,
                framewise=True,
                axiswise=True,
                width=350,
            ),
            hv.opts.NdOverlay(normalize=True, framewise=True, axiswise=True, shared_axes=False),
        )
        return curve_dmap

    def _stream_best_data(self):
        best_val = pd.DataFrame(
            {"x": [int(self.walkers.n_iters)], "best_val": [float(self.best_reward_found)]},
            columns=["x", "best_val"],
        )
        self.best_stream.send(best_val)

    def _init_reward_hist_plot(self):
        rewards = self.swarm.walkers.states.cum_rewards
        frecuency, reward = np.histogram(rewards, self.bins)
        self.reward_hist_stream = Pipe(data=(frecuency, reward))
        curve_dmap = hv.DynamicMap(hv.Histogram, streams=[self.reward_hist_stream])
        curve_dmap = curve_dmap.opts(
            hv.opts.Histogram(
                tools=["hover"],
                title="Reward distribution",
                xlabel="Reward",
                ylabel="Frequency",
                shared_axes=False,
                framewise=True,
                axiswise=True,
                width=350,
            ),
            hv.opts.NdOverlay(normalize=True, framewise=True, axiswise=True, shared_axes=False),
        )
        return curve_dmap

    def _stream_reward_hist_data(self):
        rewards = self.swarm.walkers.states.cum_rewards
        frecuency, reward = np.histogram(rewards, self.bins)
        self.reward_hist_stream.send((frecuency, reward))

    def _init_distance_hist_plot(self):
        distances = self.swarm.walkers.states.distances
        frecuency, distance = np.histogram(distances, self.bins)
        self.distance_hist_stream = Pipe(data=(frecuency, distance))
        curve_dmap = hv.DynamicMap(hv.Histogram, streams=[self.distance_hist_stream])
        curve_dmap = curve_dmap.opts(
            hv.opts.Histogram(
                tools=["hover"],
                title="Distance distribution",
                xlabel="Distance",
                ylabel="Frequency",
                shared_axes=False,
                framewise=True,
                axiswise=True,
                width=350,
            ),
            hv.opts.NdOverlay(normalize=True, framewise=True, axiswise=True, shared_axes=False),
        )
        return curve_dmap

    def _stream_distance_hist_data(self):
        distances = self.swarm.walkers.states.distances
        frecuency, distance = np.histogram(distances, self.bins)
        self.distance_hist_stream.send((frecuency, distance))

    def _init_vr_hist_plot(self):
        vrs = self.swarm.walkers.states.virtual_rewards
        frecuency, virt_reward = np.histogram(vrs, self.bins)
        self.vr_hist_stream = Pipe(data=(frecuency, virt_reward))
        curve_dmap = hv.DynamicMap(hv.Histogram, streams=[self.vr_hist_stream])
        curve_dmap = curve_dmap.opts(
            hv.opts.Histogram(
                tools=["hover"],
                title="Virtual reward distribution",
                xlabel="Virtual reward",
                ylabel="Frequency",
                shared_axes=False,
                framewise=True,
                axiswise=True,
                width=350,
                # logx=True,
            ),
            hv.opts.NdOverlay(normalize=True, framewise=True, axiswise=True, shared_axes=False),
        )
        return curve_dmap

    def _stream_vr_hist_data(self):
        virts = self.swarm.walkers.states.virtual_rewards
        frecuency, virtual_rs = np.histogram(virts, self.bins)
        self.vr_hist_stream.send((frecuency, virtual_rs))


class Plot2DSwarm(Plot1DSwarm):
    PLOT_TYPES = (
        "best",
        "reward_hist",
        "distance_hist",
        "vr_hist",
        "walkers",
        "path_best",
        "reward_2d",
        "virtual_reward_2d",
        "walker_density_2d",
    )

    def _init_walkers_plot(self):
        self.walkers_stream = Pipe(
            data=pd.DataFrame(columns=["x", "y", "reward", "virtual_reward", "will_clone"])
        )
        walkers_plot = hv.DynamicMap(hv.Scatter, streams=[self.walkers_stream])
        walkers_plot = walkers_plot.opts(
            hv.opts.Scatter(
                tools=["hover"],
                title="Walkers",
                xlabel="first dimension of data",
                ylabel="second dimension of data",
                xlim=(self.env.bounds.low[0], self.env.bounds.high[0]),
                ylim=(self.env.bounds.low[1], self.env.bounds.high[1]),
                fill_color="reward",
                size="virtual_reward",
                line_color="red",
                line_alpha="will_clone",
                line_width=1.25,
                alpha=0.7,
                cmap="viridis_r" if self.swarm.walkers.minimize else "viridis",
                shared_axes=False,
                framewise=True,
                axiswise=True,
                width=350,
            ),
            hv.opts.NdOverlay(normalize=True, framewise=True, axiswise=True, shared_axes=False),
        )
        return walkers_plot

    def _stream_walkers_data(self):
        df = pd.DataFrame(columns=["x", "y", "reward", "virtual_reward", "will_clone"])
        df["x"] = self.walkers.env_states.observs[:, 0]
        df["y"] = self.walkers.env_states.observs[:, 1]
        df["virtual_reward"] = self.walkers.states.virtual_rewards
        df["virtual_reward"] = df[["virtual_reward"]].apply(lambda x: 2.5 * (1 + x / x.max()))
        df["reward"] = self.walkers.states.cum_rewards
        df["will_clone"] = self.walkers.states.will_clone.astype(int)
        self.walkers_stream.send(df)

    def _stream_path_best_data(self):
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
        self.path_best_stream.send(best_path)

    def _init_path_best_plot(self):
        example = pd.DataFrame(
            {"x": [], "y": [], "count": [], "value": []}, columns=["x", "y", "count", "value"]
        )
        self.path_best_stream = Buffer(example.copy(), length=10000, index=False)
        best_curve_dmap = hv.DynamicMap(hv.Curve, streams=[self.path_best_stream])
        best_point_dmap = hv.DynamicMap(hv.Points, streams=[self.path_best_stream])

        best_path = (best_curve_dmap * best_point_dmap).opts(
            hv.opts.Points(
                color="count",
                width=350,
                height=350,
                line_color="black",
                size=6,
                padding=0.1,
                cmap="viridis",
                shared_axes=False,
                framewise=True,
                axiswise=True,
                xlim=(None, None),
            ),
            hv.opts.Curve(
                line_width=1,
                color="green",
                xlim=(self.env.bounds.low[0], self.env.bounds.high[0]),
                title="Evolution of best solution",
                ylim=(self.env.bounds.low[1], self.env.bounds.high[1]),
                shared_axes=False,
                framewise=True,
                axiswise=True,
            )
            # hv.opts.NdOverlay(normalize=True, framewise=True, axiswise=True, shared_axes=False),
        )
        return best_path

    def compute_plots_data(self, X, rewards, virtual_rewards):
        x = X[:, 0]
        y = X[:, 1]
        n_points = 50
        # target grid to interpolate to
        xi = np.linspace(x.min(), x.max(), n_points)
        yi = np.linspace(y.min(), y.max(), n_points)
        xx, yy = np.meshgrid(xi, yi)
        data = (X, x, y, xx, yy, rewards, virtual_rewards)
        return data

    @staticmethod
    def plot_rewards_2d(data):
        X, x, y, xx, yy, rewards, virtual_rewards, cmap = data
        zz = griddata((x, y), rewards, (xx, yy), method="linear")
        mesh = hv.QuadMesh((xx, yy, zz))
        contour = hv.operation.contours(mesh, levels=8)
        scatter = hv.Scatter((x, y))
        contour_mesh = mesh * contour * scatter
        return contour_mesh

    def _init_reward_2d_plot(self):
        cmap = "viridis_r" if self.swarm.walkers.minimize else "viridis"
        X = np.array([[512, 512], [-512, 512], [512, -512], [-512, -512]])
        init_data = self.compute_plots_data(X, np.ones(4), np.ones(4)) + (cmap,)
        self.reward_2d_stream = Pipe(data=init_data)
        reward_2d_dmap = hv.DynamicMap(self.plot_rewards_2d, streams=[self.reward_2d_stream]).opts(
            hv.opts.Contours(
                cmap=["black"],
                line_width=1,
                show_legend=False,
                shared_axes=False,
                normalize=True,
                framewise=True,
                axiswise=True,
                xlim=(-30, 30),  # (None, None),
                ylim=(-30, 30),  #
                alpha=0.9,
            ),
            hv.opts.QuadMesh(
                cmap=cmap,
                title="Interpolated reward landscape",
                bgcolor="lightgray",
                height=350,
                width=350,
                shared_axes=False,
                colorbar=True,
                normalize=True,
                framewise=True,
                axiswise=True,
                xlim=(-30, 30),  #
                ylim=(-30, 30),  #
            ),
            hv.opts.Scatter(
                fill_color="red",
                alpha=0.7,
                size=3.5,
                shared_axes=False,
                colorbar=False,
                normalize=True,
                framewise=True,
                axiswise=True,
                xlim=(-30, 30),  #
                ylim=(-30, 30),  #
            ),
        )
        return reward_2d_dmap

    def _stream_reward_2d_data(self):
        cmap = "viridis_r" if self.swarm.walkers.minimize else "viridis"
        X = self.swarm.critic.preprocess_input(
            env_states=self.walkers.env_states,
            walkers_states=self.walkers.states,
            model_states=self.walkers.model_states,
            batch_size=self.walkers.n,
        )
        rewards = self.swarm.walkers.states.cum_rewards
        virt_rewards = self.swarm.walkers.states.virtual_rewards
        data = self.compute_plots_data(X[:, :2], rewards, virt_rewards) + (cmap,)
        self.reward_2d_stream.send(data)

    @staticmethod
    def plot_virtual_reward_2d(data):
        X, x, y, xx, yy, rewards, virtual_rewards, _ = data
        zz = griddata((x, y), virtual_rewards, (xx, yy), method="linear")
        title = "Virtual reward landscape"
        mesh = hv.QuadMesh((xx, yy, zz))
        contour = hv.operation.contours(mesh, levels=8)
        scatter = hv.Scatter((x, y))
        contour_mesh = (mesh * contour * scatter).opts(
            hv.opts.Contours(
                cmap=["black"],
                line_width=1,
                show_legend=False,
                shared_axes=False,
                normalize=True,
                framewise=True,
                axiswise=True,
                xlim=(None, None),
                ylim=(None, None),
                alpha=0.9,
            ),
            hv.opts.QuadMesh(
                cmap="viridis",
                title=title,
                bgcolor="lightgray",
                height=350,
                width=350,
                shared_axes=False,
                colorbar=True,
                normalize=True,
                framewise=True,
                axiswise=True,
                xlim=(None, None),
                ylim=(None, None),
            ),
            hv.opts.Scatter(
                fill_color="red",
                alpha=0.7,
                size=3.5,
                shared_axes=False,
                colorbar=False,
                normalize=True,
                framewise=True,
                axiswise=True,
                xlim=(None, None),
                ylim=(None, None),
            ),
        )
        return contour_mesh

    def _init_virtual_reward_2d_plot(self):
        cmap = "viridis"
        X = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])
        init_data = self.compute_plots_data(X, np.ones(4), np.ones(4)) + (cmap,)
        self.virtual_reward_2d_stream = Pipe(data=init_data)
        reward_2d_dmap = hv.DynamicMap(
            self.plot_virtual_reward_2d, streams=[self.virtual_reward_2d_stream]
        ).opts(
            hv.opts.NdOverlay(normalize=True, framewise=True, axiswise=True, shared_axes=False),
        )
        return reward_2d_dmap

    def _stream_virtual_reward_2d_data(self):
        cmap = "viridis"
        X = self.swarm.critic.preprocess_input(
            env_states=self.walkers.env_states,
            walkers_states=self.walkers.states,
            model_states=self.walkers.model_states,
            batch_size=self.walkers.n,
        )
        rewards = self.swarm.walkers.states.cum_rewards
        virt_rewards = self.swarm.walkers.states.virtual_rewards
        data = self.compute_plots_data(X, rewards, virt_rewards) + (cmap,)
        self.virtual_reward_2d_stream.send(data)

    @staticmethod
    def plot_density(data):
        title = "Density distribution of walkers"
        _opts = dict(
            title=title,
            colorbar=False,
            toolbar="above",
            height=350,
            width=350,
            bandwidth=0.25,
            shared_axes=False,
            axiswise=True,
            bgcolor="lightgray",
            normalize=True,
            framewise=True,
            xlim=(None, None),
            ylim=(None, None),
            filled=True,
        )
        distribution = hv.Bivariate(data).opts(
            shared_axes=False, framewise=True
        )  # .opts(hv.opts.Bivariate(**_opts))
        return distribution

    def _init_walker_density_2d_plot(self):
        _opts = dict(
            title="Density distribution of walkers",
            colorbar=False,
            toolbar="above",
            height=350,
            width=350,
            bandwidth=0.25,
            shared_axes=False,
            axiswise=True,
            bgcolor="lightgray",
            normalize=True,
            framewise=True,
            xlim=(None, None),
            ylim=(None, None),
            filled=True,
        )
        obs = self.swarm.walkers.env_states.observs[:, :2]
        data = obs + np.random.normal(loc=0, scale=0.01, size=obs.shape)
        self.walker_density_2d_stream = Pipe(data=data)
        reward_2d_dmap = hv.DynamicMap(
            self.plot_density, streams=[self.walker_density_2d_stream]
        ).opts(
            hv.opts.Bivariate(**_opts),
            hv.opts.NdOverlay(normalize=True, framewise=True, axiswise=True, shared_axes=False),
        )
        return reward_2d_dmap

    def _stream_walker_density_2d_data(self):
        X = self.swarm.walkers.env_states.observs[:, :2]
        self.walker_density_2d_stream.send(X)


class Plot2dGrid(Plot2DSwarm):
    PLOT_TYPES = (
        "grid_2d",
        "grid_score",
        "virtual_reward_2d",
        "reward_2d",
        "walker_density_2d",
        "path_best",
        "best",
        "reward_hist",
        "distance_hist",
        "vr_hist",
        "walkers",
        "path_best",
    )

    grid_opts = dict(
        title="Memory grid values",
        line_width=None,
        tools=["hover"],
        xrotation=45,
        height=350,
        width=350,
        colorbar=True,
        cmap="viridis",
        shared_axes=False,
        bgcolor="lightgray",
        normalize=True,
        framewise=True,
        axiswise=True,
    )

    @staticmethod
    def plot_grid(data):
        xx, yy, memory_grid, _, xlim, ylim, x, y = data
        memory_vals = memory_grid.reshape(xx.shape)
        mesh = hv.QuadMesh((xx, yy, memory_vals))
        scatter_opts = dict(BasePlotSwarm.scatter_opts)
        scatter_opts["alpha"] = 0.3
        grid_opts = dict(Plot2dGrid.grid_opts)
        grid_opts["xlim"] = xlim
        grid_opts["ylim"] = ylim
        plot = (mesh * hv.Scatter((x, y))).opts(xlim=xlim, ylim=ylim)
        return plot

    def _get_grid_plot_data(self):
        X = self.swarm.critic.preprocess_input(
            env_states=self.walkers.env_states,
            walkers_states=self.walkers.states,
            model_states=self.walkers.model_states,
            batch_size=self.walkers.n,
        )
        x, y = X[:, 0], X[:, 1]
        n_points = 50
        if self.critic.bounds is None:
            self.critic.bounds = Bounds.from_array(X, scale=1.1)
        # target grid to interpolate to
        xi = np.linspace(self.critic.bounds.low[0], self.critic.bounds.high[0], n_points)
        yi = np.linspace(self.critic.bounds.low[1], self.critic.bounds.high[1], n_points)
        xx, yy = np.meshgrid(xi, yi)
        grid = np.c_[xx.ravel(), yy.ravel()]
        if self.swarm.critic.warmed:
            memory_values = self.swarm.critic.model.transform(grid)
            memory_values = np.array(
                [
                    self.swarm.critic.memory[ix[0], ix[1]].astype(np.float32)
                    for ix in memory_values.astype(int)
                ]
            )
        else:
            memory_values = np.arange(grid.shape[0])
        grid_score = self.critic.predict(grid)
        plot_bounds = self.critic.bounds.safe_margin(1.1, 1.1)
        xlim = (plot_bounds.low[0], plot_bounds.high[0])
        ylim = (plot_bounds.low[1], plot_bounds.high[1])
        return xx, yy, memory_values, grid_score, xlim, ylim, x, y

    def _init_grid_2d_plot(self):
        data = self._get_grid_plot_data()
        self.grid_2d_stream = Pipe(data=data)
        grid_opts = dict(Plot2dGrid.grid_opts)
        plot_bounds = self.critic.bounds.safe_margin(1.1, 1.1)
        xlim = (-30, 30)  # (plot_bounds.low[0], plot_bounds.high[0])
        ylim = (-30, 30)  # (plot_bounds.low[1], plot_bounds.high[1])
        grid_opts["xlim"] = xlim
        grid_opts["ylim"] = ylim
        scatter_opts = dict(BasePlotSwarm.scatter_opts)
        scatter_opts["alpha"] = 0.3
        grid_2d_dmap = hv.DynamicMap(self.plot_grid, streams=[self.grid_2d_stream]).opts(
            hv.opts.QuadMesh(**grid_opts),
            hv.opts.Scatter(**scatter_opts),
            hv.opts.NdOverlay(framewise=True, shared_axes=False),
        )
        return grid_2d_dmap

    def _stream_grid_2d_data(self):
        data = self._get_grid_plot_data()
        self.grid_2d_stream.send(data)

    def _init_grid_score_plot(self):
        data = self._get_grid_plot_data()
        xx, yy, memory_grid, grid_reward, xlim, ylim, x, y = data
        data = xx, yy, grid_reward, grid_reward, xlim, ylim, x, y
        self.grid_score_stream = Pipe(data=data)
        grid_opts = dict(Plot2dGrid.grid_opts)
        plot_bounds = self.critic.bounds.safe_margin(1.1, 1.1)
        xlim = (-30, 30)  # (plot_bounds.low[0], plot_bounds.high[0])
        ylim = (-30, 30)  # (plot_bounds.low[1], plot_bounds.high[1])
        grid_opts["xlim"] = xlim
        grid_opts["ylim"] = ylim
        scatter_opts = dict(BasePlotSwarm.scatter_opts)
        scatter_opts["alpha"] = 0.3
        scatter_opts["title"] = "Memory score"
        grid_score_dmap = hv.DynamicMap(self.plot_grid, streams=[self.grid_score_stream]).opts(
            hv.opts.QuadMesh(**grid_opts),
            hv.opts.Scatter(**scatter_opts),
            hv.opts.NdOverlay(framewise=True, shared_axes=False),
        )
        return grid_score_dmap

    def _stream_grid_score_data(self):
        data = self._get_grid_plot_data()
        xx, yy, memory_grid, grid_reward, xlim, ylim, x, y = data
        data = xx, yy, grid_reward, grid_reward, xlim, ylim, x, y
        self.grid_score_stream.send(data)


class PlotEmbeddings(Plot2dGrid):
    def _init_grid_score_plot(self):
        data = self._get_grid_plot_data()
        xx, yy, memory_grid, grid_reward, xlim, ylim, x, y = data
        data = xx, yy, grid_reward, grid_reward, xlim, ylim, x, y
        self.grid_score_stream = Pipe(data=data)
        grid_opts = dict(Plot2dGrid.grid_opts)
        plot_bounds = self.critic.bounds.safe_margin(1.1, 1.1)
        xlim = (-30, 30)  # (plot_bounds.low[0], plot_bounds.high[0])
        ylim = (-30, 30)  # (plot_bounds.low[1], plot_bounds.high[1])
        grid_opts["xlim"] = xlim
        grid_opts["ylim"] = ylim
        scatter_opts = dict(BasePlotSwarm.scatter_opts)
        scatter_opts["alpha"] = 0.3
        scatter_opts["title"] = "Memory score"
        grid_score_dmap = hv.DynamicMap(self.plot_grid, streams=[self.grid_score_stream]).opts(
            hv.opts.QuadMesh(**grid_opts),
            hv.opts.Scatter(**scatter_opts),
            hv.opts.NdOverlay(framewise=True, shared_axes=False),
        )
        return grid_score_dmap

    def _get_grid_plot_data(self):
        X = self.swarm.critic.preprocess_input(
            env_states=self.walkers.env_states,
            walkers_states=self.walkers.states,
            model_states=self.walkers.model_states,
            batch_size=self.walkers.n,
        )
        x, y = X[:, 0], X[:, 1]
        n_points = 50
        if self.critic.bounds is None:
            self.critic.bounds = Bounds.from_array(X, scale=1.1)
        # target grid to interpolate to
        xi = np.linspace(self.critic.bounds.low[0], self.critic.bounds.high[0], n_points)
        yi = np.linspace(self.critic.bounds.low[1], self.critic.bounds.high[1], n_points)
        xx, yy = np.meshgrid(xi, yi)
        grid = np.c_[xx.ravel(), yy.ravel()]
        if self.swarm.critic.warmed:
            memory_values = self.swarm.critic.model.transform(grid)
            memory_values = np.array(
                [
                    self.swarm.critic.memory[ix[0], ix[1]].astype(np.float32)
                    for ix in memory_values.astype(int)
                ]
            )
        else:
            memory_values = np.arange(grid.shape[0])
        grid_score = self.critic.predict(grid)
        plot_bounds = self.critic.bounds.safe_margin(1.1, 1.1)
        xlim = (plot_bounds.low[0], plot_bounds.high[0])
        ylim = (plot_bounds.low[1], plot_bounds.high[1])
        return xx, yy, memory_values, grid_score, xlim, ylim, x, y


class PlotKDE(Plot2DSwarm):
    PLOT_TYPES = (
        "kde",
        "reward_2d",
        "virtual_reward_2d",
        "walker_density_2d",
        "walkers",
        "path_best",
        "best",
        "reward_hist",
        "distance_hist",
        "vr_hist",
    )

    grid_opts = dict(
        title="Memory grid values",
        line_width=0,
        tools=["hover"],
        xrotation=45,
        height=350,
        width=350,
        colorbar=True,
        cmap="viridis",
        shared_axes=False,
        bgcolor="lightgray",
        normalize=True,
        framewise=True,
        axiswise=True,
    )

    @staticmethod
    def plot_kde(data):
        X, xx, yy, grid_encoded, xlim, ylim, x, y = data
        horizontal = grid_encoded.reshape(xx.shape)
        mesh = hv.QuadMesh((xx, yy, horizontal))
        scatter_opts = BasePlotSwarm.scatter_opts
        scatter_opts["alpha"] = 0.3
        grid_opts = Plot2dGrid.grid_opts
        grid_opts["xlim"] = xlim
        grid_opts["ylim"] = ylim
        plot = mesh * hv.Scatter(
            (x, y)
        )  # .opts(hv.opts.QuadMesh(**grid_opts), hv.opts.Scatter(**scatter_opts),
        # hv.opts.Layout(normalize=True, framewise=True, axiswise=True, shared_axes=False),
        # hv.opts.NdLayout(normalize=True, framewise=True, axiswise=True, shared_axes=False),
        # hv.opts.NdOverlay(normalize=True, framewise=True, axiswise=True, shared_axes=False),)

        return plot

    def _get_kde_plot_data(self):
        X = self.swarm.critic.preprocess_input(
            env_states=self.walkers.env_states,
            walkers_states=self.walkers.states,
            model_states=self.walkers.model_states,
            batch_size=self.walkers.n,
        )
        x, y = X[:, 0], X[:, 1]
        n_points = 50
        # target grid to interpolate to
        critic_bounds = self.critic.bounds
        if critic_bounds is None:
            critic_bounds = Bounds.from_array(X)
        xi = np.linspace(critic_bounds.low[0], critic_bounds.high[0], n_points)
        yi = np.linspace(critic_bounds.low[1], critic_bounds.high[1], n_points)

        xx, yy = np.meshgrid(xi, yi)
        grid = np.c_[xx.ravel(), yy.ravel()]
        scores = relativize(-self.swarm.critic.predict(grid))
        plot_bounds = critic_bounds.safe_margin(1.1, 1.1)
        xlim = (plot_bounds.low[0], plot_bounds.high[0])
        ylim = (plot_bounds.low[1], plot_bounds.high[1])
        return X, xx, yy, scores, xlim, ylim, x, y

    def _init_kde_plot(self):
        data = self._get_kde_plot_data()
        self.kde_stream = Pipe(data=data)
        grid_opts = Plot2dGrid.grid_opts
        plot_bounds = self.critic.bounds  # .safe_margin(1.1, 1.1)
        if plot_bounds is None:
            plot_bounds = Bounds.from_array(data[0])
        xlim = (plot_bounds.low[0], plot_bounds.high[0])
        ylim = (plot_bounds.low[1], plot_bounds.high[1])
        grid_opts["xlim"] = xlim
        grid_opts["ylim"] = ylim
        scatter_opts = BasePlotSwarm.scatter_opts
        scatter_opts["alpha"] = 0.3
        kde_2d_dmap = hv.DynamicMap(self.plot_kde, streams=[self.kde_stream]).opts(
            hv.opts.QuadMesh(**grid_opts),
            hv.opts.Scatter(**scatter_opts),
            hv.opts.NdOverlay(framewise=True, shared_axes=False),
        )
        return kde_2d_dmap

    def _stream_kde_data(self):
        data = self._get_kde_plot_data()
        self.kde_stream.send(data)


class LennardPlot(BasePlotSwarm):
    PLOT_TYPES = (
        "lennard_3d",
        "lennard_xy",
        "lennard_xz",
        "lennard_yz",
    )

    def __init__(self, *args, **kwargs):
        super(LennardPlot, self).__init__(*args, **kwargs)
        self._current_best = np.inf
        self.plot_data = None

    def _init_lennard_3d_plot(self):
        if self.swarm.best_found is not None:
            self.plot_data = self.swarm.best_found.reshape(-1, 3)
            x, y, z = self.plot_data[:, 0], self.plot_data[:, 1], self.plot_data[:, 2]
        else:
            _array = np.arange(10) / 10
            x, y, z = _array, _array, _array
        self.lennard_3d_stream = Pipe(data=(x, y, z))
        plot_bounds = self.env.bounds.safe_margin(1.1, 1.1)
        xlim = (plot_bounds.low[0], plot_bounds.high[0])
        ylim = (plot_bounds.low[1], plot_bounds.high[1])
        lennard_3d_dmap = hv.DynamicMap(hv.Scatter3D, streams=[self.lennard_3d_stream]).opts(
            alpha=0.9,
            size=5,
            title="Best solution found",
            xlim=xlim,
            ylim=ylim,
            # hv.opts.NdOverlay(framewise=True, shared_axes=False),
        )
        return lennard_3d_dmap

    def _stream_lennard_3d_data(self):
        # if self.swarm.best_reward_found != self._current_best:

        x, y, z = self.plot_data[:, 0], self.plot_data[:, 1], self.plot_data[:, 2]
        self.lennard_3d_stream.send((x, y, z))
        self._current_best = float(self.swarm.best_reward_found)

    def _init_lennard_3d_plot(self):
        if self.swarm.best_found is not None:
            self.plot_data = self.swarm.best_found.reshape(-1, 3)
            x, y, z = self.plot_data[:, 0], self.plot_data[:, 1], self.plot_data[:, 2]
        else:
            _array = np.arange(10) / 10
            x, y, z = _array, _array, _array
        self.lennard_3d_stream = Pipe(data=(x, y, z))
        plot_bounds = self.env.bounds.safe_margin(1.1, 1.1)
        xlim = (plot_bounds.low[0], plot_bounds.high[0])
        ylim = (plot_bounds.low[1], plot_bounds.high[1])
        lennard_3d_dmap = hv.DynamicMap(hv.Scatter3D, streams=[self.lennard_3d_stream]).opts(
            alpha=0.9,
            size=5,
            title="Best solution found",
            xlim=xlim,
            ylim=ylim,
            # hv.opts.NdOverlay(framewise=True, shared_axes=False),
        )
        return lennard_3d_dmap

    def _stream_lennard_xy_data(self):
        update = (
            self.swarm.best_reward_found < self._current_best
            if self.swarm.walkers.minimize
            else self.swarm.best_reward_found > self._current_best
        )

        if update:
            self.plot_data = self.swarm.best_found.reshape(-1, 3)
            x, y = self.plot_data[:, 0], self.plot_data[:, 1]
            self.lennard_xy_stream.send((x, y))
            self._current_best = float(self.swarm.best_reward_found)

    def _stream_lennard_xz_data(self):
        if self.swarm.best_reward_found != self._current_best:
            x, z = self.plot_data[:, 0], self.plot_data[:, 2]
            self.lennard_3d_stream.send((x, z))
            self._current_best = float(self.swarm.best_reward_found)

    def _stream_lennard_yz_data(self):
        if self.swarm.best_reward_found != self._current_best:
            y, z = self.plot_data[:, 1], self.plot_data[:, 2]
            self.lennard_3d_stream.send((y, z))
            self._current_best = float(self.swarm.best_reward_found)
