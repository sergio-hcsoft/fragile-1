import holoviews as hv
from holoviews.streams import Pipe, Buffer
import numpy as np
import pandas as pd
from umap import UMAP

from fragile.optimize.swarm import FunctionMapper


class PlotSwarm(FunctionMapper):
    def __init__(self, stream_interval: int = 10, *args, **kwargs):
        super(PlotSwarm, self).__init__(*args, **kwargs)
        self.pipe_walkers = Pipe(
            data=pd.DataFrame(columns=["x", "y", "reward", "virtual_reward", "dead"])
        )

        self.umap = UMAP(n_components=2)
        self.stream_interval = stream_interval
