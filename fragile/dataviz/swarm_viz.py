from fragile.core.walkers import StatesWalkers
from fragile.core.swarm import States, Swarm
from fragile.dataviz.swarm_stats import (
    BestReward,
    DistanceHistogram,
    DistanceLandscape,
    GridLandscape,
    InvDistanceLandscape,
    KDELandscape,
    RewardHistogram,
    RewardLandscape,
    SwarmLandscape,
    VirtualRewardHistogram,
    VirtualRewardLandscape,
    WalkersDensity,
)
from fragile.core.utils import clear_output

ALL_SWARM_TYPES = (
    GridLandscape,
    DistanceLandscape,
    RewardLandscape,
    VirtualRewardLandscape,
    WalkersDensity,
    DistanceHistogram,
    VirtualRewardHistogram,
    RewardHistogram,
    BestReward,
)

ALL_SWARM_NAMES = tuple([plot.name for plot in ALL_SWARM_TYPES])

ALL_SWARM_PLOTS = dict(zip(ALL_SWARM_NAMES, ALL_SWARM_TYPES))


class SwarmViz:
    SWARM_TYPES = (
        DistanceLandscape,
        RewardLandscape,
        VirtualRewardLandscape,
        WalkersDensity,
        DistanceHistogram,
        VirtualRewardHistogram,
        RewardHistogram,
        BestReward,
    )
    SWARM_NAMES = tuple([plot.name for plot in SWARM_TYPES])
    SWARM_PLOTS = dict(zip(SWARM_NAMES, SWARM_TYPES))

    def __init__(
        self,
        swarm: Swarm,
        display_plots="all",
        stream_interval: int = 100,
        use_embeddings: bool = True,
        margin_high=1.0,
        margin_low=1.0,
        n_points: int = 50,
    ):
        self.swarm: Swarm = swarm
        self.display_plots = self.SWARM_NAMES if display_plots == "all" else display_plots
        self.plots = self._init_plots(
            use_embeddings=use_embeddings,
            margin_low=margin_low,
            margin_high=margin_high,
            n_points=n_points,
        )
        self.stream_interval = stream_interval

    def __getattr__(self, item):
        return getattr(self.swarm, item)

    def __repr__(self):
        return self.swarm.__repr__()

    def _init_plots(self, use_embeddings, margin_low, margin_high, n_points):
        plots = {}
        for name, plot in self.SWARM_PLOTS.items():
            if issubclass(plot, SwarmLandscape):
                plots[name] = self.SWARM_PLOTS[name](
                    margin_high=margin_high,
                    n_points=n_points,
                    margin_low=margin_low,
                    use_embeddings=use_embeddings,
                )
            else:
                plots[name] = self.SWARM_PLOTS[name]()
        return plots

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
            except KeyboardInterrupt:
                break

    def run_step(self):
        self.swarm.run_step()
        if self.walkers.n_iters % self.stream_interval == 0:
            self.stream_plots()

    def stream_plots(self):
        for viz in self.plots.values():
            viz.stream_data(self)

    def plot_dmap(self, ignore: list = None):
        ignore = ignore if ignore is not None else []
        plots = [p.plot for k, p in self.plots.items() if k not in ignore]
        plot = plots[0]
        for p in plots[1:]:
            plot = plot + p
        return plot


class GridViz(SwarmViz):
    SWARM_TYPES = (
        DistanceLandscape,
        RewardLandscape,
        GridLandscape,
        VirtualRewardLandscape,
        WalkersDensity,
        DistanceHistogram,
        VirtualRewardHistogram,
        RewardHistogram,
        BestReward,
    )
    SWARM_NAMES = tuple([plot.name for plot in SWARM_TYPES])
    SWARM_PLOTS = dict(zip(SWARM_NAMES, SWARM_TYPES))


class KDEViz(SwarmViz):
    SWARM_TYPES = (
        DistanceLandscape,
        RewardLandscape,
        KDELandscape,
        VirtualRewardLandscape,
        WalkersDensity,
        DistanceHistogram,
        VirtualRewardHistogram,
        RewardHistogram,
        BestReward,
    )
    SWARM_NAMES = tuple([plot.name for plot in SWARM_TYPES])
    SWARM_PLOTS = dict(zip(SWARM_NAMES, SWARM_TYPES))


class InvDistanceViz(SwarmViz):
    SWARM_TYPES = (
        InvDistanceLandscape,
        RewardLandscape,
        DistanceLandscape,
        KDELandscape,
        VirtualRewardLandscape,
        WalkersDensity,
        DistanceHistogram,
        VirtualRewardHistogram,
        RewardHistogram,
        BestReward,
    )
    SWARM_NAMES = tuple([plot.name for plot in SWARM_TYPES])
    SWARM_PLOTS = dict(zip(SWARM_NAMES, SWARM_TYPES))
