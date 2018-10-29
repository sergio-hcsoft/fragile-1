import copy
from typing import Callable
from collections import namedtuple
import numpy as np
import torch
from fractalai.environment import Environment
from fragile.base_classes import BaseModel, BaseEnvironment, BaseSwarm, BaseWalkers


def params_to_tensors(param_dict, n_walkers: int):
    tensor_dict = {}
    copy_dict = copy.deepcopy(param_dict)
    for key, val in copy_dict.items():
        sizes = tuple([n_walkers]) + val["sizes"]
        del val["sizes"]
        tensor_dict[key] = torch.empty(sizes, **val)
    return tensor_dict


def dict_to_namedtuple(state_dict, tuple_name: str, n_walkers: int):
    tensor_dict = params_to_tensors(state_dict, n_walkers=n_walkers)
    tupletest = namedtuple(tuple_name, tensor_dict.keys())
    state_named = tupletest(**tensor_dict)
    return state_named


class States:
    def __init__(self, n_walkers: int, state_dict=None, **kwargs):
        attr_dict = (
            self.params_to_tensors(state_dict, n_walkers) if state_dict is not None else kwargs
        )
        self._names = list(attr_dict.keys())
        for key, val in attr_dict.items():
            setattr(self, key, val)
        self._n_walkers = n_walkers

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value: [torch.Tensor, np.ndarray]):
        if isinstance(value, torch.Tensor):
            setattr(self, key, value)
        elif isinstance(value, np.ndarray):
            setattr(self, key, torch.from_numpy(value))
        else:
            raise NotImplementedError(
                "You can only set attributes using torch.Tensors and " "np.ndarrays"
            )

    def __iadd__(self, other):
        self.update(other)

    def get(self, key):
        return self[key]

    def keys(self):
        return (n for n in self._names)

    def vals(self):
        return (self[name] for name in self._names)

    def items(self):
        return ((name, self[name]) for name in self._names)

    @classmethod
    def params_to_tensors(cls, param_dict, n_walkers: int):
        tensor_dict = {}
        copy_dict = copy.deepcopy(param_dict)
        for key, val in copy_dict.items():
            sizes = tuple([n_walkers]) + val["sizes"]
            del val["sizes"]
            tensor_dict[key] = torch.empty(sizes, **val)
        return tensor_dict

    @property
    def n(self):
        return self._n_walkers

    def clone(self, will_clone, compas_ix):
        for name in self.keys():
            self[name][will_clone] = self[name][compas_ix][will_clone]

    def update(self, other: "States" = None, **kwargs):
        other = other if other is not None else kwargs
        for name, val in other.items():
            self[name] = val


class TorchWalkers(BaseWalkers):
    def __init__(
        self,
        n_walkers: int,
        env_state_params: dict,
        model_state_params: dict,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
        *args,
        **kwargs
    ):
        env_state_params = self._update_env_params()
        model_state_params = self._update_env_params()
        super(TorchWalkers, self).__init__(
            n_walkers=n_walkers,
            env_state_params=env_state_params,
            model_state_params=model_state_params,
        )

        self._model_states = States(state_dict=env_state_params, n_walkers=n_walkers)
        self._env_states = States(state_dict=env_state_params, n_walkers=n_walkers)

        self.pwise_distance = torch.nn.PairwiseDistance()

        self.reward_scale = torch.tensor([reward_scale])
        self.dist_scale = torch.tensor([dist_scale])

        self.compas_ix = torch.arange(self.n)
        self.processed_rewards = torch.zeros((self.n, 1))
        self.virtual_rewards = torch.zeros((self.n, 1))
        self.distances = torch.zeros((self.n, 1))
        self.clone_probs = torch.zeros((self.n, 1))
        self.will_clone = torch.zeros((self.n, 1), dtype=torch.uint8)

        self.end_condition = torch.zeros((self.n, 1), dtype=torch.uint8)

    def get_env_states(self):
        return self._env_states

    def get_model_states(self):
        return self._model_states

    def get_obs(self):
        return self.env_states.observs

    @property
    def obs(self):
        return self.env_states.observs

    def calc_distances(self):
        with torch.no_grad():
            self.compas_ix = torch.randperm(self.n)
            self.distances = self.pwise_distance(
                self.env_states.observs, self.env_states.observs[self.compas_ix]
            )
    @staticmethod
    def _update_env_params(self, params: dict, n_walkers: int):

        default_rewards = {"rewards": {"sizes": (n_walkers, 1), "dtype": torch.float64}}
        params["reward"] = params.get("reward", default_rewards)

        default_ends = {"ends": {"sizes": (n_walkers, 1), "dtype": torch.uint8}}
        params["ends"] = params.get("ends", default_ends)

        default_death = {"death_conds": {"sizes": (n_walkers, 1), "dtype": torch.uint8}}
        params["death_conds"] = params.get("death_conds", default_death)
        return params

    @staticmethod
    def _update_model_params(self, params: dict, n_walkers: int):
        default_model_dt = {"model_dt": {"sizes": (n_walkers, 1), "dtype": torch.int64}}
        params["model_dt"] = params.get("model_dt", default_model_dt)

        default_action_dt = {"action_dt": {"sizes": (n_walkers, 1), "dtype": torch.int64}}
        params["action_dt"] = params.get("action_dt", default_action_dt)

        return params

    def normalize_rewards(self):
        def relativize(x):
            std = x.std()
            if float(std) == 0:
                return torch.ones(len(x))
            standard = (x - x.mean()) / std
            standard[standard > 0] = torch.log(1.0 + standard[standard > 0]) + 1.0
            standard[standard <= 0] = np.exp(standard[standard <= 0])
            return standard

        with torch.no_grad():
            self.processed_rewards = relativize(self.env_states.rewards)

    def calc_virtual_reward(self):
        with torch.no_grad():
            self.virtual_rewards = (
                self.processed_rewards ** self.reward_scale * self.distances ** self.dist_scale
            )

    def get_alive_compas(self):
        alives_ix = torch.arange(self.n)[self.env_states.death_cond ^ 1]
        return torch.multinomial(alives_ix, self.n, replacement=True)

    def clone_probabilities(self):
        with torch.no_grad():
            self.compas_ix = self.get_alive_compas()
            div = self.virtual_rewards
            div[div <= 0] = 1e-8
            self.clone_probs = (self.virtual_rewards[self.compas_ix] - self.virtual_rewards) / div

    def balance(self):
        self.clone_probabilities()
        self.will_clone = self.clone_probs > torch.rand(self.n)
        self._env_states.clone(will_clone=self.will_clone, compas_ix=self.compas_ix)
        self._model_states.clone(will_clone=self.will_clone, compas_ix=self.compas_ix)

    def update_states(self, env_states: States, model_states: States):
        self._env_states += env_states
        self._model_states += model_states

    def reset(self, env_states: "States", model_states: "States"):
        self.update_states(env_states=env_states, model_states=model_states)
        self.will_clone[:] = 0
        self.compas_ix[:] = torch.arange(self.n)
        self.processed_rewards[:] = torch.zeros((self.n, 1))
        self.virtual_rewards[:] = torch.zeros((self.n, 1))
        self.distances[:] = torch.zeros((self.n, 1))
        self.clone_probs[:] = torch.zeros((self.n, 1))
        self.will_clone[:] = torch.zeros((self.n, 1), dtype=torch.uint8)


class Swarm:
    def __init__(
        self,
        env: Callable,
        model: Callable,
        n_walkers: int,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
        skipframe: int=1,
    ):
        self._walkers = None
        self._model = None
        self._env = None

        self.init_swarm(env_callable=env, model_callabe=model, n_walkers=n_walkers,
                        reward_scale=reward_scale, dist_scale=dist_scale)

    @property
    def env(self):
        return self._env

    @property
    def model(self):
        return self._model

    @property
    def walkers(self):
        return self._walkers

    def init_swarm(self, env_callable: Callable, model_callabe: Callable, n_walkers: int,
                   reward_scale: float=1., dist_scale: float=1.):
        self._model = model_callabe()
        self._env = env_callable()
        model_params = self._model.get_state_params()
        env_params = self._env.get_state_params()
        self._walkers = TorchWalkers(
            env_state_params=env_params, model_state_params=model_params, n_walkers=n_walkers,
            reward_scale=reward_scale, dist_scale=dist_scale,
        )

    def init_walkers(self, model_states: "States" = None, env_states: "States" = None):
        env_sates = self.env.reset(batch_size=self.walkers.n) if env_states is None else env_states

        actions, model_states = (
            self.model.reset(batch_size=self.walkers.n,
                             env_states=env_states) if model_states is None else model_states
        )

        model_states.update(init_actions=actions)
        self.walkers.reset(env_states=env_sates, model_states=model_states)

    def run_swarm(self, model_states: "States" = None, env_states: "States" = None):
        self.init_walkers(model_states=model_states, env_states=env_states)
        while not self.walkers.calculate_end_cond():
            self.step_walkers()
            self.walkers.balance()
        return self.calculate_action()

    def step_walkers(self):
        model_states = self.walkers.get_model_states()
        env_states = self.walkers.get_env_states()
       #  model_dt, act_dt = self.model.calculate_dt(model_states, env_states)

        actions, model_states = self.model.predict(model_states, env_states)
        env_states = self.env.step(actions=actions, env_states=env_states)
        model_states.update(actions=actions)
        self.walkers.update_states(env_states=env_states, model_states=model_states)

    def calculate_action(self):
        model_states = self.walkers.get_model_states()
        init_actions = model_states.get("init_actions")
        entropy = self.walkers.get_entropy()
        sampled_actions = init_actions.unique()

        actions_dist = torch.zeros((self.model.n_actions, 1))
        for action in init_actions.unique():
            actions_dist[action] = entropy[init_actions == action].sum()
        return actions_dist

