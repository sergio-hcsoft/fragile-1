from typing import Callable

import numpy as np

from fragile.core.models import RandomContinous
from fragile.core.states import BaseStates
from fragile.core.swarm import clear_output, Swarm
from fragile.core.utils import relativize
from fragile.core.walkers import float_type, Walkers
from fragile.optimize.encoder import Encoder
from fragile.optimize.env import Function
from fragile.optimize.local_optimizer import Minimizer
from fragile.optimize.models import EncoderSampler


class MapperWalkers(Walkers):
    def __init__(self, n_vectors, timeout: int = 1000, pest_scale: float = 1, *args, **kwargs):
        self.encoder = Encoder(n_vectors=n_vectors, timeout=timeout)
        super(MapperWalkers, self).__init__(*args, **kwargs)
        self.pwise_distance = self._distances
        self._pwise_dist_module = lambda x: np.linalg.norm(x, axis=1)
        self.best_found = None
        self.best_reward_found = -1e10
        self._n_cloned_vectors = 0
        self._score_vectors = 0
        self.pest_scale = np.array([pest_scale], dtype=float_type)
        self.pests = np.zeros((self.n, 1), dtype=float_type)
        self.raw_pest = np.ones((self.n, 1), dtype=float_type)

    def __repr__(self):
        text = (
            "Best reward found: {:.5f} at position: {}, and {} "
            "cloned_vectors \n Encoder: \n {}".format(
                float(self.best_reward_found),
                self.best_found,
                self._n_cloned_vectors,
                self.encoder,
            )
        )
        return text + super(MapperWalkers, self).__repr__()

    def get_observs(self) -> np.ndarray:
        return self.observs

    def get_clone_vectors(self):
        vectors = []
        starts = self.observs[self.will_clone]
        ends = self.observs[self.compas_ix][self.will_clone]
        clones = int(self.will_clone.sum())
        for i in range(clones):
            si, ei = starts[i].copy(), ends[i].copy()
            vectors.append((si, ei))
        self._n_cloned_vectors = len(vectors)
        return vectors

    def _calculate_pests(self):
        if len(self.encoder) > 0:
            self.raw_pest = self.encoder.get_pest(self.observs).mean(1)
        self.pests = relativize(self.raw_pest)

    def calculate_virtual_reward(self):
        self._calculate_pests()
        rewards = self.processed_rewards ** self.reward_scale
        dist = self.distances ** self.dist_scale
        pest = self.pests ** -self.pest_scale
        self.virtual_rewards = rewards * dist * pest

    def balance(self):
        self.update_best()
        returned = super(MapperWalkers, self).balance()
        new_vectors = self.get_clone_vectors()
        self.encoder.update_bases(new_vectors)
        self.encoder.remove_bases(self.observs)

        return returned

    def update_best(self):
        ix = self.cum_rewards.argmax()
        best = self.observs[ix].copy()
        best_reward = float(self.cum_rewards[ix])
        if self.best_reward_found < best_reward and not bool(self.ends[ix]):
            self.best_reward_found = best_reward
            self.best_found = best

    def _distances(self, state_1, state_2):
        if True:  # len(self.encoder) < 5:#self.encoder.n_vectors:
            return self._pwise_dist_module(state_1, state_2)

        x = self.encoder.encode(state_1)
        y = self.encoder.encode(state_2)
        return torch.sum((y - x) ** 2, 1).flatten()

    def update_clone_probs(self):
        super(MapperWalkers, self).update_clone_probs()
        self.will_clone[-1] = 0

    def reset(self, env_states: BaseStates = None, model_states: BaseStates = None):
        super(MapperWalkers, self).reset(env_states=env_states, model_states=model_states)
        ix = self.cum_rewards.argmax()
        self.best_found = self.observs[ix].copy()
        self.best_reward_found = float(-1e20)


class FunctionMapper(Swarm):
    def __init__(self, plot_steps: bool = False, plot_every: int = 1e20, *args, **kwargs):
        kwargs = self.add_default_kwargs(kwargs)

        super(FunctionMapper, self).__init__(*args, **kwargs)
        self.visited_x = []
        self.visited_y = []
        self.visited_rewards = []
        self._plot_steps = plot_steps
        self.plot_every = plot_every
        self.print_i = 0
        if hasattr(self.model, "set_walkers"):
            self.model.set_walkers(self.walkers)

    @staticmethod
    def add_default_kwargs(kwargs):
        kwargs["accumulate_rewards"] = kwargs.get("accumulate_rewards", False)
        kwargs["walkers"] = kwargs.get("walkers", MapperWalkers)
        kwargs["model"] = kwargs.get("model", RandomContinous)
        return kwargs

    @classmethod
    def from_function(cls, function: Callable, shape: tuple, bounds: list = None, *args, **kwargs):
        env = Function(function=function, bounds=bounds, shape=shape)
        kwargs = cls.add_default_kwargs(kwargs)
        return FunctionMapper(env=lambda: env, *args, **kwargs)

    def _init_swarm(
        self,
        env_callable: Callable,
        model_callable: Callable,
        walkers_callable: Callable,
        n_walkers: int,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
        prune_tree: bool = True,
        *args,
        **kwargs
    ):
        super(FunctionMapper, self)._init_swarm(
            env_callable=env_callable,
            model_callable=model_callable,
            walkers_callable=walkers_callable,
            n_walkers=n_walkers,
            reward_scale=reward_scale,
            dist_scale=dist_scale,
            prune_tree=prune_tree,
            *args,
            **kwargs
        )
        self.visited_x = []
        self.visited_y = []
        self.visited_rewards = []

    def __repr__(self):

        return super(Swarm, self).__repr__()

    @property
    def encoder(self):
        return self._walkers.encoder

    def continue_optimizarion(self):
        self.print_i = 0
        while not self.walkers.calculate_end_condition():
            try:
                self.step_walkers()
                old_ids, new_ids = self.walkers.balance()
                self.prune_tree(old_ids=old_ids, new_ids=new_ids)
                if self.print_i % self.print_every == 0:
                    print(self.walkers)
                    clear_output(True)
                self.print_i += 1
            except KeyboardInterrupt as e:
                break

    def step_walkers(self):
        super(FunctionMapper, self).step_walkers()
        self.fix_best()
        # self.record_visited()
        if self._plot_steps and self.print_i % self.plot_every == 0:
            self.plot_steps()

    def fix_best(self):
        if self.walkers.best_found is not None:
            # observs = self.walkers.get_observs()
            # rewards = self.walkers.get_env_states().rewards
            self.walkers.observs[-1, :] = self.walkers.best_found.copy()
            self.walkers.rewards[-1] = float(self.walkers.best_reward_found)
            self.walkers.ends[-1] = 0

    def record_visited(self):
        observs = self.walkers.observs
        x, y = observs[:, 0].tolist(), observs[:, 1].tolist()
        rewards = self.walkers.rewards.flatten().tolist()
        self.visited_x.extend(x[:-1])
        self.visited_y.extend(y[:-1])
        self.visited_rewards.extend(rewards[:-1])

    def plot_steps(self):
        import matplotlib.pyplot as plt

        # x_vis, y_vis, rewards_vis = self.visited_x, self.visited_y, self.visited_rewards
        vals = self.walkers.observs

        x_walkers, y_walkers = vals[:, 0], vals[:, 1]
        pest = self.walkers.raw_pest.flatten().tolist()

        plt.figure(figsize=(10, 10))
        # plt.scatter(x_vis, y_vis, c=rewards_vis, cmap=plt.cm.viridis, alpha=0.1)
        plt.scatter(x_walkers, y_walkers, c=pest, cmap=plt.cm.tab20, s=30)
        # print("VALS", x_walkers, y_walkers)
        plt.colorbar()
        if self.walkers.best_found is not None:
            x_best, y_best = self.walkers.best_found[0], self.walkers.best_found[1]
            plt.scatter(x_best, y_best, color="red", marker="*", s=90)
        title = "Iteration {} {} best found {:.3f} at pos {}. N vectors: {}".format(
            self.print_i,
            self.env.function.__name__,
            float(self.walkers.best_reward_found),
            self.walkers.best_found,
            len(self.encoder),
        )

        def vector_to_arrow(v):
            x, y = v.origin[0], v.origin[1]
            dx, dy = v.end[0] - x, v.end[1] - y
            return plt.arrow(
                float(x), float(y), float(dx), float(dy), width=0.03, color="blue", alpha=0.2
            )

        for v in self.encoder.vectors:
            vector_to_arrow(v)
        min_x, max_x = self.env.bounds[0]
        min_y, max_y = self.env.bounds[1]
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)

        plt.grid()
        plt.title(title)
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        x = self.walkers.best_found.view(-1, 3)
        ax = Axes3D(fig)
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=x[:, 1], cmap=plt.cm.viridis, s=150)
        plt.grid()
        plt.show()
        plt.pause(0.01)


class LocalMapper(FunctionMapper):
    def __init__(self, minimizer: Minimizer = None, *args, **kwargs):
        super(LocalMapper, self).__init__(*args, **kwargs)
        minimizer = minimizer if minimizer is not None else Minimizer
        self.minimizer = minimizer(function=self.env)
        self.best_reward_found = -np.inf

    def minimize_best(self):
        best = self.walkers.best_found.copy()
        best_reward = float(self.walkers.best_reward_found)
        if self.best_reward_found < best_reward:
            new_best, new_best_reward = self.minimizer.minimize_point(best)
            new_best = (
                new_best
                if not np.isinf(new_best_reward) and new_best_reward > best_reward
                else best
            )
            new_best_reward = new_best_reward if not np.isinf(new_best_reward) else best_reward
            self.best_reward_found = new_best_reward
            self.walkers.best_reward_found = new_best_reward
            self.walkers.best_found = new_best

    def fix_best(self):
        if self.walkers.best_found is not None:
            self.minimize_best()
        super(LocalMapper, self).fix_best()
