import numpy as np
import torch

from fragile.core.base_classes import BaseWalkers
from fragile.core.states import BaseStates, States
from fragile.core.utils import relativize, statistics_from_array, to_tensor


# from line_profiler import profile

device_walkers = "cuda" if torch.cuda.is_available() else "cpu"

float_type = torch.float32


class Walkers(BaseWalkers):
    def __init__(
        self,
        n_walkers: int,
        env_state_params: dict,
        model_state_params: dict,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
        device=device_walkers,
        max_iters: int = 1000,
        accumulate_rewards: bool = True,
        *args,
        **kwargs
    ):

        self.device = device
        super(Walkers, self).__init__(
            n_walkers=n_walkers,
            env_state_params=env_state_params,
            model_state_params=model_state_params,
            accumulate_rewards=accumulate_rewards,
        )

        self._model_states = States(state_dict=model_state_params, n_walkers=n_walkers)
        self._env_states = States(state_dict=env_state_params, n_walkers=n_walkers)

        self.pwise_distance = torch.nn.PairwiseDistance().to(self.device)

        self.reward_scale = torch.tensor([reward_scale], dtype=float_type, device=self.device)
        self.dist_scale = torch.tensor([dist_scale], dtype=float_type, device=self.device)

        self.compas_ix = torch.arange(self.n, device=self.device)
        self.processed_rewards = torch.zeros((self.n, 1), dtype=float_type, device=self.device)
        self.virtual_rewards = torch.ones((self.n, 1), dtype=float_type, device=self.device)
        self.cum_rewards = torch.zeros((self.n, 1), dtype=float_type, device=self.device)
        self.distances = torch.zeros((self.n, 1), dtype=float_type, device=self.device)
        self.clone_probs = torch.zeros((self.n, 1), dtype=float_type, device=self.device)
        self.will_clone = torch.zeros(self.n, device=self.device, dtype=torch.uint8)

        self.end_condition = torch.zeros((self.n, 1), device=self.device, dtype=torch.uint8)
        self.alive_mask = torch.squeeze(torch.ones_like(self.will_clone))
        self.id_walkers = torch.zeros(self.n, dtype=torch.int64, device=self.device)
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
        stats = statistics_from_array(self.cum_rewards.cpu().numpy())
        text += "Total Reward: Mean: {:.3f}, Std: {:.3f}, Max: {:.3f} Min: {:.3f}\n".format(*stats)
        stats = statistics_from_array(self.virtual_rewards.cpu().numpy())
        text += "Virtual Rewards: Mean: {:.3f}, Std: {:.3f}, Max: {:.3f} Min: {:.3f}\n".format(
            *stats
        )
        stats = statistics_from_array(self.distances.cpu().numpy())
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
                k, shape, *statistics_from_array(v.cpu().numpy())
            )
            string += new_str
        return string

    @property
    def observs(self) -> torch.Tensor:
        try:
            return self._env_states.observs.to(self.device)
        except Exception as e:
            if not hasattr(self._env_states, "observs"):
                raise AttributeError(
                    "observs is not a valid attribute of env_states, please make "
                    "sure it exists before calling self.obs and make sure it is "
                    "an instance of torch.Tensor"
                )
            elif not isinstance(self._env_states.observs, torch.Tensor):
                raise TypeError(
                    "The type of self.env_states.observs is {} instead of a "
                    "torch.Tensor".format(type(self._env_states.observs))
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

    def get_obs(self) -> torch.Tensor:
        return self.observs

    def update_end_condition(self, ends: [torch.Tensor, np.ndarray]):
        if isinstance(ends, np.ndarray):
            ends = torch.from_numpy(ends.astype(np.uint8)).to(device_walkers)
        self.end_condition = ends

    def calc_end_condition(self) -> bool:
        all_dead = np.array(self.end_condition.cpu()).sum() == self.n
        max_iters = self.n_iters > self.max_iters
        self.n_iters += 1
        return all_dead or max_iters

    def calc_distances(self):
        with torch.no_grad():
            self.compas_ix = torch.randperm(self.n, dtype=torch.int64, device=self.device)
            self.distances = relativize(self.pwise_distance(
                self.observs.view(self.n, -1).float(),
                self.observs[self.compas_ix].view(self.n, -1).float(),
            ).view(-1, 1)).view(-1, 1)

    def normalize_rewards(self):
        with torch.no_grad():
            self.processed_rewards = relativize(self.cum_rewards, device=self.device).view(-1, 1)

    def calc_virtual_reward(self):
        with torch.no_grad():
            rewards = self.processed_rewards.float() ** self.reward_scale.float()
            dist = self.distances.float() ** self.dist_scale.float()
            virtual_reward = rewards * dist
            self.virtual_rewards = to_tensor(virtual_reward, dtype=float_type)

    def get_alive_compas(self):
        self.alive_mask = torch.squeeze(self.end_condition) ^ 1
        if torch.all(self.alive_mask == self.alive_mask[0]):
            return torch.arange(self.n).to(self.device)

        compas = torch.multinomial(self.alive_mask.float(), self.n, replacement=True)

        return torch.squeeze(compas)

    def update_clone_probs(self):
        with torch.no_grad():
            if torch.all(self.virtual_rewards == self.virtual_rewards[0]):
                probs = torch.ones(self.n, device=self.device, dtype=float_type) / float(self.n)
                return probs
            self.compas_ix = self.get_alive_compas()
            div = torch.clamp(self.virtual_rewards.float(), 1e-8)
            # This value can be negative!!
            clone_probs = (self.virtual_rewards[self.compas_ix] - self.virtual_rewards) / div
            self.clone_probs = clone_probs

    # @profile
    def balance(self):
        old_ids = set(self.id_walkers.cpu().numpy().astype(int).tolist())
        self.normalize_rewards()
        self.calc_distances()
        self.calc_virtual_reward()
        self.update_clone_probs()
        rands = torch.rand(self.n).view(-1, 1).to(device_walkers)
        self.will_clone = torch.squeeze(self.clone_probs > rands)
        dead_ix = torch.arange(self.n)[torch.squeeze(self.end_condition)]
        self.will_clone[dead_ix] = 1

        self.cum_rewards[self.will_clone] = self.cum_rewards[self.compas_ix][self.will_clone]
        self.id_walkers[self.will_clone] = self.id_walkers[self.compas_ix][self.will_clone]

        self._env_states.clone(will_clone=self.will_clone, compas_ix=self.compas_ix)
        self._model_states.clone(will_clone=self.will_clone, compas_ix=self.compas_ix)
        new_ids = set(self.id_walkers.cpu().numpy().astype(int).tolist())
        return old_ids, new_ids

    def update_states(self, env_states: BaseStates = None, model_states: BaseStates = None):
        if isinstance(env_states, BaseStates):
            self._env_states.update(env_states)
        if hasattr(env_states, "rewards"):
            self.accumulate_rewards(env_states.rewards)
        if isinstance(model_states, BaseStates):
            self._model_states.update(model_states)

    def accumulate_rewards(self, rewards: [torch.Tensor, np.ndarray]):
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards)
        if self._accumulate_rewards:
            self.cum_rewards = self.cum_rewards + rewards.to(self.device).float()
        else:
            self.cum_rewards = rewards.to(self.device).float().reshape(-1, 1)

    def update_ids(self, walkers_ids: np.ndarray):
        self.id_walkers[:] = torch.from_numpy(walkers_ids).to(self.device)

    def reset(self, env_states: "BaseStates" = None, model_states: "BaseStates" = None):
        self.update_states(env_states=env_states, model_states=model_states)
        self.will_clone[:] = torch.zeros(self.n, dtype=torch.uint8)
        self.compas_ix[:] = torch.arange(self.n).to(self.device)
        self.processed_rewards[:] = torch.zeros((self.n, 1), dtype=float_type, device=self.device)
        self.cum_rewards[:] = torch.zeros((self.n, 1), dtype=float_type, device=self.device)
        self.virtual_rewards[:] = torch.ones((self.n, 1), dtype=float_type, device=self.device)
        self.distances[:] = torch.zeros((self.n, 1), dtype=float_type, device=self.device)
        self.clone_probs[:] = torch.zeros((self.n, 1), dtype=float_type, device=self.device)
        self.will_clone[:] = torch.zeros(self.n, dtype=torch.uint8)
        self.alive_mask[:] = torch.squeeze(torch.ones_like(self.will_clone))
        self.id_walkers[:] = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.n_iters = 0
