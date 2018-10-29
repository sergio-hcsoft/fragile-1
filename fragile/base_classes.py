from typing import Iterable


class BaseEnvironment:
    @property
    def n_actions(self):
        raise NotImplementedError

    def step(self, actions, env_states):
        raise NotImplementedError

    def reset(self, batch_size: int = 1):
        raise NotImplementedError


class BaseModel:
    def __init__(self, *args, **kwargs):
        self.actor = None
        self.critic = None
        self.world_emb = None
        self.simulation = None
        self._init_modules(*args, **kwargs)

    @property
    def n_actions(self):
        raise NotImplementedError

    def reset(self, batch_size: int = 1):
        raise NotImplementedError

    def predict(self, model_states, env_states):
        """
        Given a state return the next action to be taken.
        :param state:
        :return:
        """
        raise NotImplementedError

    def actor_pred(self, *args, **kwargs):
        raise NotImplementedError

    def critic_pred(self, *args, **kwargs):
        raise NotImplementedError

    def world_emb_pred(self, *args, **kwargs):
        raise NotImplementedError

    def simulation_pred(self, *args, **kwargs):
        raise NotImplementedError

    def calculate_skipframe(self):
        raise NotImplementedError

    def _init_actor(self, *args, **kwargs):
        return None

    def _init_critic(self, *args, **kwargs):
        return None

    def _init_world_emb(self, *args, **kwargs):
        return None

    def _init_simulation(self, *args, **kwargs):
        return None

    def _init_modules(self, *args, **kwargs):
        self._init_world_emb(*args, **kwargs)
        self._init_critic(*args, **kwargs)
        self._init_actor(*args, **kwargs)
        self._init_simulation(*args, **kwargs)


class BaseWalkers:
    def __init__(
        self, n_walkers: int, env_state_params: dict, model_state_params: dict, *args, **kwargs
    ):
        self.model_state_params = model_state_params
        self.env_state_params = env_state_params
        self.n_walkers = n_walkers
        self.rewards = None
        self.death_cond = None
        self.observs = None

    @property
    def n(self):
        return self.n_walkers

    @property
    def env_states(self):
        raise NotImplementedError

    @property
    def model_states(self):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def calc_distances(self):
        raise NotImplementedError

    def calc_scores(self):
        raise NotImplementedError

    def clone(self, clone_ix, will_clone):
        raise NotImplementedError

    def get_alive_compas(self):
        raise NotImplementedError

    def update_obs(self, *args, **kwargs):
        raise NotImplementedError

    def update_env_states(self, *env_states, **kwargs):
        raise NotImplementedError

    def update_model_states(self, *env_states, **kwargs):
        raise NotImplementedError


class BaseSwarm:
    def __init__(
        self,
        model: BaseModel,
        env: BaseEnvironment,
        walkers: BaseWalkers,
        max_samples: int,
        reward_scale: float = 1,
        dist_scale: float = 1,
        *args,
        **kwargs
    ):
        self._model = model
        self._env = env
        self.walkers = walkers
        self.max_samples = max_samples
        self.reward_scale = reward_scale
        self.dist_scale = dist_scale
        self.actions = None
        self.skipframes = None

    @property
    def model(self):
        return self._model

    @property
    def env(self):
        return self._env

    def run_swarm(self, *args, **kwargs):
        self.reset(*args, **kwargs)
        while not self._end_condition(*args, **kwargs):
            self.step_walkers(*args, **kwargs)
            self.balance(*args, **kwargs)
            self._update_end_condition(*args, **kwargs)

    def reset(self, model_state=None, env_state=None, *args, **kwargs):
        env_states = (
            env_state if env_state is not None else self.env.reset(batch_size=self.walkers.n)
        )
        model_states = (
            model_state if model_state is None else self.model.reset(batch_size=self.walkers.n)
        )
        self.walkers.update_states(env_states=env_states, model_states=model_states)

    def perturbate_walkers(self, env_states, model_states):
        actions, *new_model_states = self.model.predict(model_states)
        observs, rewards, ends, deaths, *new_env_states = self.env.step(
            actions=actions, states=env_states
        )
        return new_env_states, new_model_states, actions, observs, rewards, ends, deaths

    def step_walkers(self):
        env_states, model_states = self.walkers.env_states, self.walkers.model_states
        skipframes = self.model.calculate_skipframes()
        for i in range(skipframes.max()):
            env_states, model_states, actions, observs, rewards, ends, deaths = self.perturbate_walkers(
                env_states=env_states, model_states=model_states
            )

        self.walkers.update_states(
            env_states=env_states,
            model_states=model_states,
            rewards=rewards,
            ends=ends,
            deaths=deaths,
            actions=self.actions,
            observs=observs,
        )

    def calculate_virtual_reward(self):
        distances = self.walkers.calc_distances()
        scores = self.walkers.calc_scores()
        virt_r = distances ** self.dist_scale * scores ** self.reward_scale
        return virt_r

    def balance(self, *args, **kwargs):
        ix_compas = self.walkers.get_alive_compas()
        virtual_reward = self.calculate_virtual_reward()
        clone_prob = self._get_clone_prob(virtual_reward=virtual_reward, ix_compas=ix_compas)
        will_clone = self._get_clone_index(clone_prob)
        self.walkers.clone(will_clone=will_clone, clone_ix=ix_compas)

    def _get_clone_prob(self, virtual_reward, ix_compas):
        raise NotImplementedError

    def _get_clone_index(self, virtual_reward, ix_compas):
        raise NotImplementedError

    def _end_condition(self, *args, **kwargs):
        pass

    def _update_end_condition(self, *args, **kwargs):
        pass
