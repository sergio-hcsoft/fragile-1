import pytest

import holoviews

from fragile.core.models import NormalContinuous
from fragile.dataviz import AtariViz, LandscapeViz, Summary, SwarmViz, SwarmViz1D
from fragile.optimize.benchmarks import EggHolder
from fragile.optimize.swarm import FunctionMapper
from tests.core.test_swarm import TestSwarm, create_atari_swarm


holoviews.extension("bokeh")


def create_eggholder_swarm():
    def gaussian_model(env):
        # Gaussian of mean 0 and std of 10, adapted to the environment bounds
        return NormalContinuous(scale=10, loc=0.0, bounds=env.bounds)

    swarm = FunctionMapper(
        env=EggHolder, model=gaussian_model, n_walkers=300, max_iters=750, start_same_pos=True,
    )
    return swarm


def create_summary():
    swarm = create_eggholder_swarm()
    return Summary(swarm, stream_interval=2)


def create_viz_1d():
    swarm = create_eggholder_swarm()
    return SwarmViz1D(swarm, stream_interval=2)


def create_landscape_viz():
    swarm = create_eggholder_swarm()
    return LandscapeViz(swarm, stream_interval=2)


def create_atari_viz_default():
    swarm = create_atari_swarm()
    return AtariViz(swarm, stream_interval=2)


def create_atari_viz_all():
    swarm = create_atari_swarm()
    return AtariViz(swarm, stream_interval=2, display_plots="all")


def create_swarmviz():
    swarm = create_eggholder_swarm()
    return SwarmViz(swarm, stream_interval=2)


PLOTS = {}


swarm_dict = {
    "summary": create_summary,
    "swarm_viz_1d": create_viz_1d,
    "create_landscape_viz": create_landscape_viz,
    "atari_viz_default": create_atari_viz_default,
    "atari_viz_all": create_atari_viz_all,
    "swarm_viz": create_swarmviz,
}
swarm_names = list(swarm_dict.keys())


class TestSwarmVisualizations(TestSwarm):
    @pytest.fixture(params=swarm_names)
    def swarm(self, request):
        swarm_viz = swarm_dict.get(request.param)()
        PLOTS[request.param] = swarm_viz.plot()
        return swarm_viz

    @pytest.fixture(params=swarm_names)
    def swarm_with_score(self, request):
        return None

    def test_score_gets_higher(self, swarm_with_score):
        pass
