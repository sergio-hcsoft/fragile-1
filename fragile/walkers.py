import torch
import numpy as np
from fragile.states import States
from fragile.base_classes import BaseWalkers
from fragile.utils import relativize
device_walkers = "cuda" if torch.cuda.is_available() else "cpu"


class Walkers(BaseWalkers):
    def __init__(
        self,
        n_walkers: int,
        env_state_params: dict,
        model_state_params: dict,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
        device=device_walkers,
        max_iters: int=1000,
        *args,
        **kwargs
    ):

        self.device = device
        super(Walkers, self).__init__(
            n_walkers=n_walkers,
            env_state_params=env_state_params,
            model_state_params=model_state_params,
        )

        self._model_states = States(state_dict=model_state_params, n_walkers=n_walkers)
        self._env_states = States(state_dict=env_state_params, n_walkers=n_walkers)

        self.pwise_distance = torch.nn.PairwiseDistance().to(self.device)

        self.reward_scale = torch.tensor([reward_scale], device=self.device)
        self.dist_scale = torch.tensor([dist_scale], device=self.device)

        self.compas_ix = torch.arange(self.n, device=self.device)
        self.processed_rewards = torch.zeros((self.n, 1), device=self.device)
        self.virtual_rewards = torch.ones((self.n, 1), device=self.device)
        self.cum_rewards = torch.zeros((self.n, 1), device=self.device)
        self.distances = torch.zeros((self.n, 1), device=self.device)
        self.clone_probs = torch.zeros((self.n, 1), device=self.device)
        self.will_clone = torch.zeros(self.n, device=self.device, dtype=torch.uint8)

        self.end_condition = torch.zeros((self.n, 1), device=self.device, dtype=torch.uint8)
        self.alive_mask = torch.squeeze(torch.ones_like(self.will_clone))
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
            raise type(e)(str(e) + " Error at Walkers.__getattr__%s" % msg).with_traceback(
                sys.exc_info()[2]
            )

    def __repr__(self) -> str:
        text = "{}\n".format(self.__class__.__name__)
        text += "Env {}\n".format(self._env_states.__repr__())
        text += "Model {}\n".format(self._model_states.__repr__())
        return text

    @property
    def obs(self) -> torch.Tensor:
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
    def env_states(self) -> States:
        return self._env_states

    @property
    def model_states(self) -> States:
        return self._model_states

    def get_env_states(self) -> States:
        return self.env_states

    def get_model_states(self) -> States:
        return self.model_states

    def get_obs(self) -> torch.Tensor:
        return self.obs

    def update_end_condition(self, ends: [torch.Tensor, np.ndarray]):
        if isinstance(ends, np.ndarray):
            ends = torch.from_numpy(ends.astype(np.uint8)).to(device_walkers)
        self.end_condition = ends

    def calculate_end_cond(self) -> bool:
        all_dead = np.array(self.end_condition.cpu()).all()
        max_iters = self.n_iters > self.max_iters
        self.n_iters += 1
        return all_dead or max_iters

    def calc_distances(self):
        with torch.no_grad():
            self.compas_ix = torch.randperm(self.n, dtype=torch.int64, device=self.device)
            self.distances = self.pwise_distance(self.obs, self.obs[self.compas_ix])

    def normalize_rewards(self):
        with torch.no_grad():
            self.processed_rewards = relativize(self.cum_reward, device=self.device)

    def calc_virtual_reward(self):
        with torch.no_grad():
            self.virtual_rewards = (
                self.processed_rewards ** self.reward_scale * self.distances ** self.dist_scale
            )

    def get_alive_compas(self):
        self.alive_mask = torch.squeeze(self.end_condition) ^ 1
        compas = torch.multinomial(self.alive_mask.float(), self.n, replacement=True)
        return torch.squeeze(compas)

    def update_clone_probs(self):
        with torch.no_grad():
            if torch.all(self.virtual_rewards == self.virtual_rewards[0]):
                probs = torch.ones(self.n, device=self.device, dtype=torch.float32) / float(self.n)
                return probs
            self.compas_ix = self.get_alive_compas()
            div = torch.clamp(self.virtual_rewards.float(), 1e-8)
            # This value can be negative!!
            clone_probs = (self.virtual_rewards[self.compas_ix] - self.virtual_rewards) / div
            self.clone_probs = clone_probs

    def balance(self):
        self.update_clone_probs()
        rands = torch.rand(self.n).view(-1, 1).to(device_walkers)
        self.will_clone = torch.squeeze(self.clone_probs > rands)
        self._env_states.clone(will_clone=self.will_clone, compas_ix=self.compas_ix)
        self._model_states.clone(will_clone=self.will_clone, compas_ix=self.compas_ix)

    def update_states(self, env_states: States = None, model_states: States = None):
        if isinstance(env_states, States):
            self._env_states.update(env_states)
            if hasattr(env_states, "rewards"):
                self.accumulate_rewards(env_states.rewards)
        if isinstance(model_states, States):
            self._model_states.update(model_states)

    def accumulate_rewards(self, rewards: [torch.Tensor, np.ndarray]):
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards)
        self.cum_rewards = self.cum_rewards + rewards.to(self.device)

    def reset(self, env_states: "States" = None, model_states: "States" = None):
        self.update_states(env_states=env_states, model_states=model_states)
        self.will_clone[:] = torch.zeros(self.n, dtype=torch.uint8)
        self.compas_ix[:] = torch.arange(self.n).to(self.device)
        self.processed_rewards[:] = 0.
        self.cum_rewards[:] = 0.
        self.virtual_rewards[:] = 1.0  # torch.ones((self.n, 1))
        self.distances[:] = 0.
        self.clone_probs[:] = 0.
        self.will_clone[:] = torch.zeros(self.n, dtype=torch.uint8)
        self.alive_mask[:] = torch.squeeze(torch.ones_like(self.will_clone))
        self.n_iters = 0
