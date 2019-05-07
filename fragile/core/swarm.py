from typing import Callable

import torch

try:
    from IPython.core.display import clear_output
except ImportError:
    clear_output = lambda x: x

# from line_profiler import profile

from fragile.core.base_classes import (
    BaseEnvironment,
    BaseModel,
    BaseStates,
    BaseSwarm,
    BaseWalkers,
)
from fragile.core.tree import Tree


class Swarm(BaseSwarm):
    @property
    def env(self) -> BaseEnvironment:
        return self._env

    @property
    def model(self) -> BaseModel:
        return self._model

    @property
    def walkers(self) -> BaseWalkers:
        return self._walkers

    def _init_swarm(
        self,
        env_callable: Callable,
        model_callabe: Callable,
        walkers_callable: Callable,
        n_walkers: int,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
        *args,
        **kwargs
    ):
        self._env: BaseEnvironment = env_callable()
        self._model: BaseModel = model_callabe(self._env)

        model_params = self._model.get_params_dict()
        env_params = self._env.get_params_dict()
        self._walkers: BaseWalkers = walkers_callable(
            env_state_params=env_params,
            model_state_params=model_params,
            n_walkers=n_walkers,
            reward_scale=reward_scale,
            dist_scale=dist_scale,
            *args,
            **kwargs
        )

        self.tree = Tree()

    def init_walkers(self, model_states: BaseStates = None, env_states: BaseStates = None):
        env_sates = self.env.reset(batch_size=self.walkers.n) if env_states is None else env_states

        actions, model_states = (
            self.model.reset(batch_size=self.walkers.n, env_states=env_states)
            if model_states is None
            else model_states
        )

        model_states.update(init_actions=actions)
        self.walkers.reset(env_states=env_sates, model_states=model_states)
        self.tree.reset(
            env_state=self.walkers.env_states,
            model_state=self.walkers.model_states,
            reward=env_sates.rewards[0],
        )

    # @profile
    def run_swarm(
        self,
        model_states: BaseStates = None,
        env_states: BaseStates = None,
        print_every: int = 1e100,
    ):
        self.init_walkers(model_states=model_states, env_states=env_states)
        print_i = 0
        while not self.walkers.calc_end_condition():
            self.step_walkers()
            old_ids, new_ids = self.walkers.balance()
            self.prune_tree(old_ids=old_ids, new_ids=new_ids)
            if print_i % print_every == 0:
                print(self.walkers)
                clear_output(True)
            print_i += 1

        return self.calculate_action()

    # @profile
    def step_walkers(self):
        model_states = self.walkers.get_model_states()
        states_ids = self.walkers.id_walkers.cpu().numpy().copy().astype(int).flatten().tolist()
        env_states = self.walkers.get_env_states()
        act_dt, model_states = self.model.calculate_dt(model_states, env_states)

        actions, model_states = self.model.predict(
            env_states=env_states, model_states=model_states, batch_size=self.walkers.n
        )
        env_states = self.env.step(actions=actions, env_states=env_states, n_repeat_action=act_dt)
        model_states.update(actions=actions)

        self.walkers.update_states(env_states=env_states, model_states=model_states)
        self.walkers.update_end_condition(env_states.ends)
        walker_ids = self.tree.add_states(
            parent_ids=states_ids,
            env_states=self.walkers.env_states,
            model_states=self.walkers.model_states,
            cum_rewards=self.walkers.cum_rewards.cpu().numpy().copy().flatten(),
        )
        self.walkers.update_ids(walker_ids)

    def prune_tree(self, old_ids, new_ids):
        dead_leaves = old_ids - new_ids
        for leaf_id in dead_leaves:
            self.tree.prune_branch(leaf_id=leaf_id)

    def calculate_action(self):
        return
        model_states = self.walkers.get_model_states()
        init_actions = model_states.get("init_actions")
        entropy = self.walkers.get_entropy()

        actions_dist = torch.zeros((self.model.n_actions, 1))
        for action in init_actions.unique():
            actions_dist[action] = entropy[init_actions == action].sum()
        return actions_dist
