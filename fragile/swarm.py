import torch
from typing import Callable
from fragile.states import States
from fragile.walkers import Walkers

# from line_profiler import profile


class Swarm:
    def __init__(
        self,
        env: Callable,
        model: Callable,
        n_walkers: int,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
        skipframe: int = 1,
        *args,
        **kwargs
    ):
        self._walkers = None
        self._model = None
        self._env = None
        self.skipframe = skipframe

        self.init_swarm(
            env_callable=env,
            model_callabe=model,
            n_walkers=n_walkers,
            reward_scale=reward_scale,
            dist_scale=dist_scale,
            *args,
            **kwargs
        )

    @property
    def env(self):
        return self._env

    @property
    def model(self):
        return self._model

    @property
    def walkers(self):
        return self._walkers

    def init_swarm(
        self,
        env_callable: Callable,
        model_callabe: Callable,
        n_walkers: int,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
        *args,
        **kwargs
    ):
        self._env = env_callable()
        self._model = model_callabe(self._env.n_actions)

        model_params = self._model.get_params_dict()
        env_params = self._env.get_params_dict()
        self._walkers = Walkers(
            env_state_params=env_params,
            model_state_params=model_params,
            n_walkers=n_walkers,
            reward_scale=reward_scale,
            dist_scale=dist_scale,
            *args,
            **kwargs
        )

    def init_walkers(self, model_states: "States" = None, env_states: "States" = None):
        env_sates = self.env.reset(batch_size=self.walkers.n) if env_states is None else env_states

        actions, model_states = (
            self.model.reset(batch_size=self.walkers.n, env_states=env_states)
            if model_states is None
            else model_states
        )

        model_states.update(init_actions=actions)
        self.walkers.reset(env_states=env_sates, model_states=model_states)

    # @profile
    def run_swarm(self, model_states: "States" = None, env_states: "States" = None):
        self.init_walkers(model_states=model_states, env_states=env_states)
        while not self.walkers.calculate_end_cond():
            self.step_walkers()
            self.walkers.balance()

        return self.calculate_action()

    # @profile
    def step_walkers(self):
        # model_states = self.walkers.get_model_states()
        env_states = self.walkers.get_env_states()
        # model_dt, act_dt = self.model.calculate_dt(model_states, env_states)

        actions = self.model.predict(env_states, batch_size=self.walkers.n)
        env_states = self.env.step(
            actions=actions, env_states=env_states, n_repeat_action=self.skipframe
        )
        # model_states.update(actions=actions)
        self.walkers.update_states(env_states=env_states)  # , model_states=model_states)
        self.walkers.update_end_condition(env_states.ends)

    def calculate_action(self):
        return
        model_states = self.walkers.get_model_states()
        init_actions = model_states.get("init_actions")
        entropy = self.walkers.get_entropy()
        sampled_actions = init_actions.unique()

        actions_dist = torch.zeros((self.model.n_actions, 1))
        for action in init_actions.unique():
            actions_dist[action] = entropy[init_actions == action].sum()
        return actions_dist
