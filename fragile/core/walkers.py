import numpy as np

from fragile.core.base_classes import BaseWalkers
from fragile.core.states import BaseStates, States
from fragile.core.utils import relativize, statistics_from_array


#import line_profiler

float_type = np.float32


class Walkers(BaseWalkers):
    def __init__(
        self,
        n_walkers: int,
        env_state_params: dict,
        model_state_params: dict,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
        max_iters: int = 1000,
        accumulate_rewards: bool = True,
        *args,
        **kwargs
    ):

        super(Walkers, self).__init__(
            n_walkers=n_walkers,
            env_state_params=env_state_params,
            model_state_params=model_state_params,
            accumulate_rewards=accumulate_rewards,
        )

        self._model_states = States(state_dict=model_state_params, n_walkers=n_walkers)
        self._env_states = States(state_dict=env_state_params, n_walkers=n_walkers)

        self.pwise_distance = lambda x: np.linalg.norm(x, axis=1)

        self.reward_scale = reward_scale
        self.dist_scale = dist_scale

        self.compas_ix = np.arange(self.n)
        self.processed_rewards = np.zeros(self.n, dtype=float_type)
        self.virtual_rewards = np.ones(self.n, dtype=float_type)
        self.cum_rewards = np.zeros(self.n, dtype=float_type)
        self.distances = np.zeros(self.n, dtype=float_type)
        self.clone_probs = np.zeros(self.n, dtype=float_type)
        self.will_clone = np.zeros(self.n, dtype=np.bool_)

        self.end_condition = np.zeros(self.n, dtype=np.bool_)
        self.alive_mask = np.ones_like(self.will_clone)
        self.id_walkers = np.zeros(self.n, dtype=np.int64)
        self.n_iters = 0
        self.max_iters = max_iters

    def __getattr__(self, item):
        if hasattr(super(Walkers, self), item):
            return super(Walkers, self).__getattribute__(item)
        elif hasattr(self._env_states, item):
            return self._env_states.__getattribute__(item)
        elif hasattr(self._model_states, item):
            return self._model_states.__getattribute__(item)
        try:
            return super(Walkers, self).__getattribute__(item)
        except Exception as e:
            import sys

            msg = "\nAttribute {} is not in the class nor in its internal states".format(item)
            raise type(e)(str(e) + " Error at Walkers.__getattr__: %s\n" % msg).with_traceback(
                sys.exc_info()[2]
            )

    def __repr__(self) -> str:
        text = self.print_stats()
        text += "Env: {}\n".format(self.__repr_state(self._env_states))
        text += "Model {}\n".format(self.__repr_state(self._model_states))
        return text

    def print_stats(self) -> str:
        text = "{} iteration {}\n".format(self.__class__.__name__, self.n_iters)
        stats = statistics_from_array(self.cum_rewards)
        text += "Total Reward: Mean: {:.3f}, Std: {:.3f}, Max: {:.3f} Min: {:.3f}\n".format(*stats)
        stats = statistics_from_array(self.virtual_rewards)
        text += "Virtual Rewards: Mean: {:.3f}, Std: {:.3f}, Max: {:.3f} Min: {:.3f}\n".format(
            *stats
        )
        stats = statistics_from_array(self.distances)
        text += "Distances: Mean: {:.3f}, Std: {:.3f}, Max: {:.3f} Min: {:.3f}\n".format(*stats)

        text += "Dead walkers: {:.2f}% Cloned: {:.2f}%\n".format(
            100 * self.end_condition.sum() / self.n, 100 * self.will_clone.sum() / self.n
        )
        return text

    @staticmethod
    def __repr_state(state):
        string = "\n"
        for k, v in state.items():
            if k in ["observs", "states"]:
                continue
            shape = v.shape if hasattr(v, "shape") else None
            new_str = "{} shape {} Mean: {:.3f}, Std: {:.3f}, Max: {:.3f} Min: {:.3f}\n".format(
                k, shape, *statistics_from_array(v)
            )
            string += new_str
        return string

    @property
    def observs(self) -> np.ndarray:
        try:
            return self._env_states.observs
        except Exception as e:
            if not hasattr(self._env_states, "observs"):
                raise AttributeError(
                    "observs is not a valid attribute of env_states, please make "
                    "sure it exists before calling self.obs and make sure it is "
                    "an instance of np.ndarray"
                )
            raise e

    @property
    def env_states(self) -> BaseStates:
        return self._env_states

    @property
    def model_states(self) -> BaseStates:
        return self._model_states

    def get_env_states(self) -> BaseStates:
        return self.env_states

    def get_model_states(self) -> BaseStates:
        return self.model_states

    def get_obs(self) -> np.ndarray:
        return self.observs

    def update_end_condition(self, ends: np.ndarray):
        self.end_condition = ends

    def calc_end_condition(self) -> bool:
        all_dead = np.array(self.end_condition).sum() == self.n
        max_iters = self.n_iters > self.max_iters
        self.n_iters += 1
        return all_dead or max_iters

    def calc_distances(self):
        self.compas_ix = np.random.permutation(np.arange(self.n))
        distances = np.linalg.norm(
            self.observs.reshape(self.n, -1) - self.observs[self.compas_ix].reshape(self.n, -1),
            axis=1,
        ).flatten()
        self.distances = relativize(distances)

    def normalize_rewards(self):
        self.processed_rewards = relativize(self.cum_rewards)

    def calc_virtual_reward(self):
        rewards = self.processed_rewards ** self.reward_scale
        dist = self.distances ** self.dist_scale
        self.virtual_rewards = rewards * dist

    def get_alive_compas(self):
        self.alive_mask = np.logical_not(self.end_condition)
        if not self.alive_mask.any():
            return np.arange(self.n)
        compas_ix = np.arange(len(self.alive_mask))[self.alive_mask]
        compas = np.random.choice(compas_ix, self.n, replace=True)
        return compas

    def update_clone_probs(self):

        if (self.virtual_rewards.flatten() == self.virtual_rewards[0]).all():
            probs = np.ones(self.n, dtype=float_type) / float(self.n)
            return probs
        self.compas_ix = self.get_alive_compas()
        div = np.clip(self.virtual_rewards, 1e-8, np.inf)
        # This value can be negative!!
        clone_probs = (self.virtual_rewards[self.compas_ix] - self.virtual_rewards) / div
        self.clone_probs = clone_probs

    # @profile
    def balance(self):
        old_ids = set(self.id_walkers.astype(int).tolist())
        self.normalize_rewards()
        self.calc_distances()
        self.calc_virtual_reward()
        self.update_clone_probs()
        rands = np.random.random(self.n)
        self.will_clone = self.clone_probs > rands
        dead_ix = np.arange(self.n)[self.end_condition.flatten()]
        self.will_clone[dead_ix] = 1

        self.cum_rewards[self.will_clone] = self.cum_rewards[self.compas_ix][self.will_clone]
        self.id_walkers[self.will_clone] = self.id_walkers[self.compas_ix][self.will_clone]

        self._env_states.clone(will_clone=self.will_clone, compas_ix=self.compas_ix)
        self._model_states.clone(will_clone=self.will_clone, compas_ix=self.compas_ix)
        new_ids = set(self.id_walkers.astype(int).tolist())
        return old_ids, new_ids

    def update_states(self, env_states: BaseStates = None, model_states: BaseStates = None):
        if isinstance(env_states, BaseStates):
            self._env_states.update(env_states)
        if hasattr(env_states, "rewards"):
            self.accumulate_rewards(env_states.rewards)
        if isinstance(model_states, BaseStates):
            self._model_states.update(model_states)

    def accumulate_rewards(self, rewards: np.ndarray):
        if self._accumulate_rewards:
            self.cum_rewards = self.cum_rewards + rewards
        else:
            self.cum_rewards = rewards

    def update_ids(self, walkers_ids: np.ndarray):
        self.id_walkers[:] = walkers_ids

    def reset(self, env_states: "BaseStates" = None, model_states: "BaseStates" = None):
        self.update_states(env_states=env_states, model_states=model_states)
        self.will_clone[:] = np.zeros(self.n, dtype=np.bool_)
        self.compas_ix[:] = np.arange(self.n)
        self.processed_rewards[:] = np.zeros(self.n, dtype=float_type)
        self.cum_rewards[:] = np.zeros(self.n, dtype=float_type)
        self.virtual_rewards[:] = np.ones(self.n, dtype=float_type)
        self.distances[:] = np.zeros(self.n, dtype=float_type)
        self.clone_probs[:] = np.zeros(self.n, dtype=float_type)
        self.alive_mask[:] = np.ones_like(self.will_clone)
        self.id_walkers[:] = np.zeros(self.n, dtype=np.int64)
        self.n_iters = 0
