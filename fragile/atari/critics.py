from holoviews.streams import Pipe, Buffer
from streamz.dataframe import DataFrame
from streamz import Stream
import holoviews as hv
import hvplot.pandas
import hvplot.streamz
import numpy as np
import pandas as pd

from fragile.core.base_classes import States, BaseCritic
from fragile.core.utils import resize_frame, relativize


class MontezumaGrid(BaseCritic):
    NUM_ROOMS = 50
    MAX_COORDINATES = (320, 160)

    def __init__(
        self,
        shape=(16, 16),
        scale: float = 1.0,
        recover_n: int = 1,
        decrease_n: int = 10,
        forget_val: float = 0.05,
    ):
        self.scale = scale
        self.grid_shape = shape
        self.recover_n = recover_n
        self.decrease_n = decrease_n
        self.memory = np.zeros(shape + (self.NUM_ROOMS,), dtype=np.int64)  # * 1000
        self.x_mod = self.MAX_COORDINATES[0] // self.grid_shape[0]
        self.y_mod = self.MAX_COORDINATES[1] // self.grid_shape[1]
        self.n_iter = 0
        self.stream = Stream()
        self._cols = [str(i) for i in range(shape[0])]
        self._index = [str(i) for i in range(shape[1])]
        example = pd.DataFrame(self.memory[:, :, 0], columns=self._cols, index=self._index)
        self.buffer_df = DataFrame(stream=self.stream, example=example)
        self.forget_val = forget_val

    def calculate(
        self,
        batch_size: int = None,
        model_states: States = None,
        env_states: States = None,
        walkers_states: "StatesWalkers" = None,
    ) -> np.ndarray:
        """
        Calculate the target time step values.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the model data.
            env_states: States corresponding to the environment data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Array containing the target time step.

        """
        self.n_iter += 1
        x = (env_states.observs[:, -3] // self.x_mod).astype(int)
        y = (env_states.observs[:, -2] // self.y_mod).astype(int)
        rooms = env_states.observs[:, -1].astype(int)
        self.update_memory(x, y, rooms)
        vals = self._calculate_values(x, y, rooms)
        walkers_states.update(critic_score=vals)
        return vals

    def update_memory(self, xs, ys, rooms):
        # for x, y, room in zip(xs, ys, rooms):
        self.memory[xs, ys, rooms] += 1
        self.memory = np.clip(self.memory - self.forget_val, 0, np.inf)

    def _calculate_values(self, x, y, rooms):
        prob_no_walker = 1 - self.memory[x, y, rooms] / self.n_iter / len(x)
        # score = 1 / (1 + np.log(1 + vals))
        # self.stream.emit(pd.DataFrame(self.memory[:, :, rooms[0]],
        #                              columns=self._cols, index=self._index))
        # print(self.memory[:, :, rooms[0]], x, y, rooms)
        return relativize(prob_no_walker ** 2) ** self.scale

    def reset(self, batch_size: int = 1, model_states: States = None, *args, **kwargs) -> States:
        """
        Restart the DtSampler and reset its internal state.

        Args:
            batch_size: Number of elements in the first dimension of the model \
                        States data.
            model_states: States corresponding to model data. If provided the \
                          model will be reset to this state.
            args: Additional arguments not related to model data.
            kwargs: Additional keyword arguments not related to model data.

        Returns:
            States containing the information of the current state of the \
            model (after the reset).

        """
        self.n_iter = 0
        n_walkers = model_states.n if model_states is not None else batch_size
        self.memory = np.zeros(self.grid_shape + (self.NUM_ROOMS,), dtype=np.int32)
        return np.ones(n_walkers)

    def plot_grid(self):
        return self.buffer_df.hvplot(kind="bar")  # heatmap()

    def update(self, *args, **kwargs):
        pass
