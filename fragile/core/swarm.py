import copy
from typing import Callable, List


try:
    from IPython.core.display import clear_output
except ImportError:

    def clear_output(**kwargs):
        pass


# from line_profiler import profile

from fragile.core.base_classes import BaseEnvironment, BaseModel, BaseSwarm
from fragile.core.states import States
from fragile.core.tree import Tree
from fragile.core.walkers import Walkers


class Swarm(BaseSwarm):
    """
        The Swarm is in charge of performing a fractal evolution process.

        It contains the necessary logic to use an Environment, a Model, and a \
        Walkers instance to run the Swarm evolution algorithm.
    """

    @property
    def env(self) -> BaseEnvironment:
        return self._env

    @property
    def model(self) -> BaseModel:
        return self._model

    @property
    def walkers(self) -> Walkers:
        return self._walkers

    def _init_swarm(
        self,
        env_callable: Callable,
        model_callable: Callable,
        walkers_callable: Callable,
        n_walkers: int,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
        prune_tree: bool = True,
        use_tree: bool = False,
        *args,
        **kwargs
    ):
        """
        Initialize and set up all the necessary internal variables to run the swarm.
        This process involves instantiating the Swarm, the Environment and the
        model.

        Args:
            env_callable: A function that returns an instance of an
                :class:`fragile.Environment`.
            model_callable: A function that returns an instance of a
                :class:`fragile.Model`.
            walkers_callable: A function that returns an instance of
                :class:`fragile.Walkers`.
            n_walkers: Number of walkers of the swarm.
            reward_scale: Virtual reward exponent for the reward score.
            dist_scale:Virtual reward exponent for the distance score.

        Returns:
            None
        """
        self._env: BaseEnvironment = env_callable()
        self._model: BaseModel = model_callable(self._env)

        model_params = self._model.get_params_dict()
        env_params = self._env.get_params_dict()
        self._walkers: Walkers = walkers_callable(
            env_state_params=env_params,
            model_state_params=model_params,
            n_walkers=n_walkers,
            reward_scale=reward_scale,
            dist_scale=dist_scale,
            *args,
            **kwargs
        )

        self.tree = Tree() if use_tree else None
        self._prune_tree = prune_tree
        self._use_tree = use_tree
        self.epoch = 0

    def reset(
        self, walkers_states: States = None, model_states: States = None, env_states: States = None
    ):
        env_sates = self.env.reset(batch_size=self.walkers.n) if env_states is None else env_states

        actions, model_states = (
            self.model.reset(batch_size=self.walkers.n, env_states=env_states)
            if model_states is None
            else model_states
        )

        model_states.update(init_actions=actions)
        self.walkers.reset(env_states=env_sates, model_states=model_states)
        if self._use_tree:
            self.tree.reset(
                env_state=self.walkers.env_states,
                model_state=self.walkers.model_states,
                reward=env_sates.rewards[0],
            )

    # @profile
    def run_swarm(
        self, model_states: States = None, env_states: States = None, print_every: int = 1e100
    ):
        self.reset(model_states=model_states, env_states=env_states)
        self.epoch = 0
        while not self.walkers.calculate_end_condition():
            try:
                self.step_walkers()
                old_ids, new_ids = self.walkers.balance()
                self.prune_tree(old_ids=set(old_ids.tolist()), new_ids=set(new_ids.tolist()))
                if self.epoch % print_every == 0:
                    print(self.walkers)
                    clear_output(True)
                self.epoch += 1
            except KeyboardInterrupt as e:
                break

    # @profile
    def step_walkers(self):
        """Make the walkers undergo a random perturbation process in the swarm \
        Environment.
        """
        model_states = self.walkers.model_states
        env_states = self.walkers.env_states

        states_ids = (
            copy.deepcopy(self.walkers.states.id_walkers).astype(int).flatten().tolist()
            if self._use_tree
            else None
        )
        act_dt, model_states = self.model.calculate_dt(model_states, env_states)

        actions, model_states = self.model.predict(
            env_states=env_states, model_states=model_states
        )
        env_states = self.env.step(actions=actions, env_states=env_states, n_repeat_action=act_dt)
        model_states.update(actions=actions)

        self.walkers.update_states(
            env_states=env_states, model_states=model_states, end_condition=env_states.ends
        )
        self.update_tree(states_ids)

    def update_tree(self, states_ids: List[int]):
        if self._use_tree:
            walker_ids = self.tree.add_states(
                parent_ids=states_ids,
                env_states=self.walkers.env_states,
                model_states=self.walkers.model_states,
                cum_rewards=copy.deepcopy(self.walkers.states.cum_rewards).flatten(),
            )
            self.walkers.update_states(id_walkers=walker_ids)

    def prune_tree(self, old_ids, new_ids):
        if self._prune_tree:
            dead_leaves = old_ids - new_ids
            for leaf_id in dead_leaves:
                self.tree.prune_branch(leaf_id=leaf_id)

    """def calculate_action(self):
        return
        model_states = self.walkers.get_model_states()
        init_actions = model_states.get("init_actions")
        entropy = self.walkers.get_entropy()

        actions_dist = np.zeros((self.model.n_actions, 1))
        for action in init_actions.unique():
            actions_dist[action] = entropy[init_actions == action].sum()
        return actions_dist"""
