from collections import deque
from typing import Callable

from scipy.interpolate import griddata
import holoviews as hv
from holoviews import opts
from holoviews.streams import Pipe
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import KBinsDiscretizer
from streamz import Stream
from umap import UMAP

from fragile.core.bounds import Bounds
from fragile.core.base_classes import BaseCritic, States
from fragile.core.utils import relativize


class Memory(BaseCritic):
    def __init__(
        self,
        scale: float = 1,
        model=None,
        warmup_epochs: int = 100,
        maxlen: int = None,
        refresh: int = 1000,
        *args,
        **kwargs
    ):
        self.refresh = refresh
        self._model_args = args
        self._model_kwargs = kwargs
        self.model_class: Callable = model
        self.warmup_epochs = warmup_epochs
        self.scale = scale
        self.maxlen = maxlen
        self.warmed = False
        self.buffer = deque([], maxlen=self.maxlen)
        self.epoch = 0
        self.model = self._init_model()

    def ready_to_train(self) -> bool:
        raise NotImplementedError

    def train_model(
        self,
        env_states: States,
        walkers_states: "StatesWalkers",
        batch_size: int = None,
        model_states: States = None,
    ) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray):
        raise NotImplementedError

    def preprocess_input(
        self,
        env_states: States,
        walkers_states: "StatesWalkers",
        batch_size: int = None,
        model_states: States = None,
    ) -> np.ndarray:
        return env_states.observs

    def get_score(
        self,
        env_states: States,
        walkers_states: "StatesWalkers",
        batch_size: int = None,
        model_states: States = None,
    ) -> np.ndarray:
        processed_data = self.preprocess_input(
            env_states=env_states,
            walkers_states=walkers_states,
            model_states=model_states,
            batch_size=batch_size,
        )
        probs = self.predict(processed_data)
        score = relativize(-probs)
        return score ** self.scale

    def _init_model(self):
        if self.model_class is None:
            return
        return self.model_class(*self._model_args, **self._model_kwargs)

    def reset(self, batch_size: int = 1, model_states: States = None, *args, **kwargs) -> States:
        """
        Restart the Memory and reset its internal state.

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
        self.warmed = False
        self.buffer = deque([], maxlen=self.maxlen)
        self.epoch = 0
        self.model = self._init_model()

    def calculate(
        self,
        env_states: States,
        walkers_states: "StatesWalkers",
        batch_size: int = None,
        model_states: States = None,
    ) -> np.ndarray:
        self.epoch += 1
        self.update(
            env_states=env_states,
            walkers_states=walkers_states,
            model_states=model_states,
            batch_size=batch_size,
        )
        warmup_not_finished = self.epoch < self.warmup_epochs and not self.warmed
        if warmup_not_finished:
            # The critic does nothing
            score = np.ones(env_states.n)
            walkers_states.update(critic_score=score)
            return score
        elif self.ready_to_train():
            self.train_model(
                env_states=env_states,
                walkers_states=walkers_states,
                model_states=model_states,
                batch_size=batch_size,
            )
        scores = self.get_score(
            env_states=env_states,
            walkers_states=walkers_states,
            batch_size=batch_size,
            model_states=model_states,
        )
        walkers_states.update(critic_score=scores)
        return scores

    def update(
        self,
        env_states: States,
        walkers_states: "StatesWalkers",
        batch_size: int = None,
        model_states: States = None,
    ) -> None:
        self.buffer.append(env_states.observs)


class GridRepulsion(Memory):
    def __init__(
        self,
        embedding: Callable = None,
        warmup_epochs: int = 10,
        refresh: int = 100000,
        scale: float = 1.0,
        bounds_margin: float = 1.0,
        n_bins: int = 10,
        out_of_bins: int = 100,
        forget_val: float = 0.0,
        strategy="uniform",
        *args,
        **kwargs
    ):
        """

        Args:
            warmup:
            refresh:
            *args:
            **kwargs:
        """
        model = KBinsDiscretizer
        self.embedding = embedding() if embedding is not None else None
        self.n_bins = n_bins
        self.scale = scale
        self.bounds_margin = bounds_margin
        self.max_out_of_bins = out_of_bins
        self.forget_val = forget_val
        self.bounds = None
        self.memory = None
        self.out_of_bins = 0

        super(GridRepulsion, self).__init__(
            model=model,
            warmup_epochs=warmup_epochs,
            refresh=refresh,
            n_bins=n_bins,
            encode="ordinal",  # The code relays on this encoding mode
            strategy=strategy,
            *args,
            **kwargs
        )

    def reset(self, batch_size: int = 1, model_states: States = None, *args, **kwargs) -> States:
        super(GridRepulsion, self).reset(
            batch_size=batch_size, model_states=model_states, *args, **kwargs
        )
        self.out_of_bins = 0

    def ready_to_train(self) -> bool:
        time_to_refresh = self.epoch % self.refresh == 0 and self.warmed
        warmup_finished = self.epoch == self.warmup_epochs and not self.warmed
        too_many_out_of_bins = self.out_of_bins > self.max_out_of_bins
        return time_to_refresh or warmup_finished or too_many_out_of_bins

    def train_compression(
        self,
        env_states: States,
        walkers_states: "StatesWalkers",
        batch_size: int = None,
        model_states: States = None,
    ) -> np.ndarray:
        X = np.concatenate(self.buffer, axis=0)
        embeddings = self.embedding.fit_transform(X) if self.embedding is not None else X
        self.bounds = Bounds.from_array(embeddings, scale=self.bounds_margin)
        return embeddings

    def train_model(
        self,
        env_states: States,
        walkers_states: "StatesWalkers",
        batch_size: int = None,
        model_states: States = None,
    ) -> None:

        embeddings = self.train_compression(
            env_states=env_states,
            model_states=model_states,
            walkers_states=walkers_states,
            batch_size=batch_size,
        )
        self.model = self.model.fit(embeddings)
        self.memory = np.zeros((self.n_bins,) * embeddings.shape[1])
        self.warmed = True
        self.epoch = 0
        self.buffer = deque([], maxlen=self.maxlen)
        self.out_of_bins = 0

    def _update_memory(self, indexes: np.ndarray):
        for ix in indexes:
            self.memory[tuple(ix)] += 1
        # self.memory[indexes[:, 0], indexes[:, 1]] += 1
        self.memory = np.clip(self.memory - self.forget_val, 0, np.inf)

    def preprocess_input(
        self,
        env_states: States,
        walkers_states: "StatesWalkers",
        batch_size: int = None,
        model_states: States = None,
    ) -> np.ndarray:
        X = env_states.observs
        try:
            processed_data = self.embedding.transform(X)
        except Exception as e:
            processed_data = X
        if self.bounds is None:
            self.bounds = Bounds.from_array(X, scale=1.1)
        points_out = np.logical_not(self.bounds.points_in_bounds(processed_data))
        alive_points = np.logical_not(walkers_states.end_condition)
        self.out_of_bins += np.logical_and(alive_points, points_out).sum()
        # return self.embedding.transform(X) if self.embedding is not None else X
        return processed_data

    def predict(self, X: np.ndarray):
        if not self.warmed:
            return np.ones(X.shape[0])
        indexes = self.model.transform(X).astype(int)
        self._update_memory(indexes)
        scores = np.array([self.memory[tuple(ix)].astype(np.float32) for ix in indexes])
        prob_no_walker = 1 - scores / self.epoch / self.n_bins
        return prob_no_walker

    def get_score(
        self,
        env_states: States,
        walkers_states: "StatesWalkers",
        batch_size: int = None,
        model_states: States = None,
    ) -> np.ndarray:
        processed_data = self.preprocess_input(
            env_states=env_states,
            walkers_states=walkers_states,
            model_states=model_states,
            batch_size=batch_size,
        )
        probs = self.predict(processed_data)
        score = relativize(-probs)
        return score ** self.scale


class GaussianRepulssion(GridRepulsion):
    def __init__(
        self,
        embedding=None,
        forget_val: float = 0.0,
        warmup_epochs: int = 100,
        refresh: int = 1000,
        scale: float = 1.0,
        model=KernelDensity,
        *args,
        **kwargs
    ):
        """

        Args:
            warmup:
            refresh:
            *args:
            **kwargs:
        """
        self.scale = scale
        self.embedding = embedding() if embedding is not None else None
        self.scale = scale
        self.bounds_margin = 1.0
        self.max_out_of_bins = np.inf
        self.forget_val = forget_val
        self.memory = None
        self.bounds = None
        self.out_of_bins = 0
        self.n_bins = 0

        super(GridRepulsion, self).__init__(
            model=model, warmup_epochs=warmup_epochs, refresh=refresh, *args, **kwargs
        )

    def ready_to_train(self) -> bool:
        time_to_refresh = self.epoch % self.refresh == 0 and self.warmed
        warmup_finished = self.epoch == self.warmup_epochs and not self.warmed
        return time_to_refresh or warmup_finished

    def __train_model(
        self,
        env_states: States,
        walkers_states: "StatesWalkers",
        batch_size: int = None,
        model_states: States = None,
    ) -> None:
        buffer = np.concatenate(self.buffer, axis=0)
        self.model.fit(np.concatenate([buffer, env_states.observs], axis=0))
        self.warmed = True
        self.buffer = deque([], maxlen=self.maxlen)

    def __preprocess_input(
        self,
        env_states: States,
        walkers_states: "StatesWalkers",
        batch_size: int = None,
        model_states: States = None,
    ) -> np.ndarray:
        return env_states.observs

    def predict(self, X: np.ndarray):
        if not self.warmed:
            return np.ones(X.shape[0])
        return self.model.score_samples(X)


class EmbeddingRepulsion(GridRepulsion):
    def __init__(self, embedding: Callable = None, *args, **kwargs):
        embedding = self.default_umap() if embedding is None else embedding
        super(EmbeddingRepulsion, self).__init__(embedding=embedding, *args, **kwargs)

    @staticmethod
    def default_umap():
        _umap = UMAP(n_components=2)
        return lambda: _umap


class __BinRepulssion(BaseCritic):
    def __init__(
        self,
        warmup: int = 100,
        refresh_rate: int = 10000,
        n_bins=100,
        strategy="uniform",
        scale: float = 1.0,
        forget_val: float = 0.05,
        bounds_scale: float = 1.0,
        *args,
        **kwargs
    ):
        self.refresh_rate = refresh_rate
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.warmup = warmup
        self._warmed = False
        self.umap = UMAP(*args, **kwargs)
        self.bins: KBinsDiscretizer = KBinsDiscretizer(
            n_bins=n_bins, encode=encode, strategy=strategy
        )
        self.memory = None
        self.buffer = []
        self._epoch = 1
        self.out_of_bins = 0
        self.scale = scale
        self.forget_val = forget_val
        self.bounds = Bounds(high=np.ones(2) * 100, low=np.zeros(2))
        self.bounds_scale = bounds_scale

    def train_model(self):
        X = np.concatenate(self.buffer, axis=0)
        embeddings = self.umap.fit_transform(X)
        self.bins = self.bins.fit(embeddings)
        self.bounds = Bounds.from_array(embeddings, scale=self.bounds_scale)
        self.memory = np.zeros((self.n_bins,) * embeddings.shape[1])
        self._warmed = True
        self._epoch = 0
        self.buffer = []
        self.out_of_bins = 0

    def calculate(
        self,
        env_states: States,
        walkers_states: "StatesWalkers",
        batch_size: int = None,
        model_states: States = None,
    ) -> None:
        self._epoch += 1
        self.buffer.append(env_states.observs)
        warmup_not_finished = self._epoch < self.warmup and not self._warmed
        ready_to_train = (
            self.out_of_bins > self.refresh_rate
            and self._warmed
            or self._epoch == self.warmup
            and not self._warmed
        )
        if warmup_not_finished:
            score = np.ones(env_states.n)
            walkers_states.update(critic_score=score)
            return
        elif ready_to_train:
            self.train_model()
        embeddings = self.umap.transform(env_states.observs)
        indexes = self.bins.transform(embeddings).astype(int)
        points_out = np.logical_not(self.bounds.points_in_bounds(embeddings))
        alive_points = np.logical_not(walkers_states.end_condition)
        self.out_of_bins += np.logical_and(alive_points, points_out).sum()
        self._update_memory(indexes)
        scores = self._get_score(indexes)

        walkers_states.update(critic_score=scores)
        return scores

    def __repr__(self):
        return str("Memory: %s" % self.memory)

    def reset(self, batch_size: int = 1, model_states: States = None, *args, **kwargs) -> States:
        self.buffer = []
        self._epoch = 1
        self.bins = KBinsDiscretizer(
            n_bins=self.n_bins, encode=self.encode, strategy=self.strategy
        )
        self._warmed = False

    def update(self, *args, **kwargs):
        pass

    def _update_memory(self, indexes: np.ndarray):
        for ix in indexes:
            self.memory[tuple(ix)] += 1
        # self.memory[indexes[:, 0], indexes[:, 1]] += 1
        self.memory = np.clip(self.memory - self.forget_val, 0, np.inf)

    def _get_score(self, indexes):
        scores = np.array([self.memory[tuple(ix)].astype(np.float32) for ix in indexes])
        prob_no_walker = 1 - scores / self._epoch / self.n_bins
        scores = relativize(prob_no_walker) ** self.scale
        return scores


class __BinDebugger(__BinRepulssion):
    def __init__(self, stream_interval: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_interval = stream_interval
        X = np.array([[512, 512], [-512, 512], [512, -512], [-512, -512]])
        init_data = self.compute_plots_data(X, np.ones(4), np.ones(4))
        self.pipe = Pipe(data=init_data)
        self.dmap = hv.DynamicMap(self._make_stream_plot, streams=[self.pipe]).opts(
            shared_axes=False, normalize=True, shared_datasource=False, framewise=True,
        )

    def stream_plots(self, walkers_states, env_states):
        X = env_states.observs
        rewards = walkers_states.cum_rewards
        virtual_rewards = walkers_states.virtual_rewards
        data = self.compute_plots_data(X, rewards, virtual_rewards)
        self.pipe.send(data)

    def compute_plots_data(self, X, rewards, virtual_rewards):
        x = X[:, 0]
        y = X[:, 1]
        n_points = 50
        # target grid to interpolate to
        xi = np.linspace(x.min(), x.max(), n_points)
        yi = np.linspace(y.min(), y.max(), n_points)
        xx, yy = np.meshgrid(xi, yi)
        grid = np.c_[xx.ravel(), yy.ravel()]
        if self._warmed:
            grid_encoded = self.bins.transform(grid)
            grid_encoded = np.array(
                [self.memory[tuple(ix)].astype(np.float32) for ix in grid_encoded.astype(int)]
            )

        else:
            grid_encoded = np.arange(grid.shape[0])

        data = (X, x, y, xx, yy, rewards, virtual_rewards, grid_encoded)
        return data

    def _make_stream_plot(self, data):
        X, x, y, xx, yy, rewards, virtual_rewards, grid_encoded = data
        return self.make_plot(X, x, y, xx, yy, rewards, virtual_rewards, grid_encoded)

    @classmethod
    def make_plot(cls, X, x, y, xx, yy, rewards, virtual_rewards, grid_encoded):
        # scatter = cls.make_scatter(x, y)
        density = cls.plot_density(X, None, filled=True, cmap="viridis")
        grid = cls.plot_grid(xx, yy, grid_encoded, None)
        rewards_plot = cls.plot_rewards(x, y, xx, yy, rewards, None, cmap="viridis_r")
        vr_plot = cls.plot_rewards(x, y, xx, yy, virtual_rewards, None).opts(
            title="Virtual rewards landscape", framewise=True, shared_axes=False, normalize=True,
        )
        return (
            (density + rewards_plot + grid + vr_plot)
            .opts(framewise=True, shared_axes=False, axiswise=True, normalize=True)
            .cols(2)
        )

    @staticmethod
    def plot_rewards(x, y, xx, yy, rewards, scatter=None, cmap="viridis"):
        zz = griddata((x, y), rewards, (xx, yy), method="linear")
        title = "Interpolated reward landscape"
        mesh = hv.QuadMesh((xx, yy, zz)).opts(
            cmap=cmap,
            title=title,
            bgcolor="lightgray",
            height=400,
            width=500,
            shared_axes=False,
            colorbar=True,
            normalize=True,
            framewise=True,
            axiswise=True,
            xlim=(x.min(), x.max()),
            ylim=(None, None),
        )

        contour = hv.operation.contours(mesh, levels=8).opts(
            cmap="copper",
            line_width=2,
            height=500,
            width=500,
            legend_position="top",
            shared_axes=False,
            normalize=True,
            framewise=True,
            axiswise=True,
            xlim=(None, None),
            ylim=(None, None),
        )
        contour_mesh = (mesh * contour).opts(
            normalize=True,
            framewise=True,
            axiswise=True,
            xlim=(None, None),
            ylim=(None, None),
            shared_axes=False,
        )

        return mesh if scatter is None else contour_mesh * scatter

    @staticmethod
    def make_scatter(x, y):
        scatter = hv.Scatter((x, y))
        return scatter.opts(
            shared_axes=False,
            color="red",
            size=3,
            alpha=0.4,
            normalize=True,
            framewise=True,
            xlim=(None, None),
            ylim=(None, None),
            axiswise=True,
        )

    @staticmethod
    def plot_density(data, scatter=None, **kwargs):
        title = "Density distribution of walkers"
        _opts = dict(
            title=title,
            colorbar=False,
            toolbar="above",
            height=400,
            width=500,
            bandwidth=0.1,
            shared_axes=False,
            axiswise=True,
            bgcolor="lightgray",
            normalize=True,
            framewise=True,
            xlim=(None, None),
            ylim=(None, None),
            **kwargs
        )
        sco = dict(
            normalize=True,
            framewise=True,
            xlim=(None, None),
            axiswise=True,
            ylim=(None, None),
            shared_axes=False,
        )
        distribution = hv.Bivariate(data).opts(**_opts)
        return (
            distribution if scatter is None else (distribution * scatter.opts(**sco)).opts(**sco)
        )

    @staticmethod
    def plot_grid(xx, yy, grid_encoded, scatter=None):
        horizontal = grid_encoded.reshape(xx.shape)
        mesh = hv.QuadMesh((xx, yy, horizontal))
        title = "Memory grid values"
        mesh = mesh.opts(
            title=title,
            line_width=0,
            tools=["hover"],
            xrotation=45,
            height=400,
            width=500,
            colorbar=True,
            cmap="viridis",
            shared_axes=False,
            bgcolor="lightgray",
            normalize=True,
            framewise=True,
            axiswise=True,
            xlim=(None, None),
            ylim=(None, None),
        )

        return mesh if scatter is None else mesh * scatter

    def calculate(
        self,
        env_states: States,
        walkers_states: "StatesWalkers",
        batch_size: int = None,
        model_states: States = None,
    ) -> None:
        super(BinDebugger, self).calculate(
            env_states=env_states,
            walkers_states=walkers_states,
            batch_size=batch_size,
            model_states=model_states,
        )
        # return self.stream_plots(walkers_states, env_states)
        if self._epoch % self.stream_interval == 0:
            self.stream_plots(walkers_states, env_states)
