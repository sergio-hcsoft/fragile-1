import copy
from typing import Callable, List


try:
    from IPython.core.display import clear_output
except ImportError:

    def clear_output(**kwargs):
        """If not using jupyter notebook do nothing."""
        pass


import line_profiler


from fragile.core.base_classes import BaseEnvironment, BaseModel, BaseStateTree, BaseSwarm
from fragile.core.states import States
from fragile.core.walkers import StatesWalkers, Walkers


class Swarm(BaseSwarm):
    """
    The Swarm is in charge of performing a fractal evolution process.

    It contains the necessary logic to use an Environment, a Model, and a \
    Walkers instance to run the Swarm evolution algorithm.
    """

    def __init__(self, walkers: Callable = Walkers, *args, **kwargs):
        super(Swarm, self).__init__(walkers=walkers, *args, **kwargs)

    def __repr__(self):
        return self.walkers.__repr__()

    @property
    def env(self) -> BaseEnvironment:
        """All the simulation code (problem specific) will be handled here."""
        return self._env

    @property
    def model(self) -> BaseModel:
        """
        All the policy and random perturbation code (problem specific) will \
        be handled here.
        """
        return self._model

    @property
    def walkers(self) -> Walkers:
        """
        Access the :class:`Walkers` in charge of implementing the FAI \
        evolution process.
        """
        return self._walkers

    @property
    def best_found(self):
        return self.walkers.states.best_found

    @property
    def best_reward_found(self):
        return self.walkers.states.best_reward_found

    @property
    def critic(self):
        return self._walkers.critic

    def _init_swarm(
        self,
        env_callable: Callable,
        model_callable: Callable,
        walkers_callable: Callable,
        n_walkers: int,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
        tree: Callable = None,
        prune_tree: bool = True,
        *args,
        **kwargs
    ):
        """
        Initialize and set up all the necessary internal variables to run the swarm.

        This process involves instantiating the Swarm, the Environment and the \
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
            dist_scale: Virtual reward exponent for the distance score.
            use_tree: If True, initialize a :class:`Tree` to store the \
                      visited states.
            prune_tree: If `use_tree` is False it has no effect. If true, \
                       store in the :class:`Tree` the past history of alive walkers.

        Returns:
            None.

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
        self._use_tree = tree is not None
        self.tree: BaseStateTree = tree() if self._use_tree else None
        self._prune_tree = prune_tree
        self.epoch = 0

    def reset(
        self,
        walkers_states: StatesWalkers = None,
        model_states: States = None,
        env_states: States = None,
    ):
        """
        Reset a :class:`fragile.Walkers` and clear the isnternal data to start a \
        new search process.

        Args:
            model_states: States that define the initial state of the environment.
            env_states: States that define the initial state of the model.
            walkers_states: States that define the internal states of the walkers.

        """
        env_sates = self.env.reset(batch_size=self.walkers.n) if env_states is None else env_states

        model_states = (
            self.model.reset(batch_size=self.walkers.n, env_states=env_states)
            if model_states is None
            else model_states
        )

        model_states.update(init_actions=model_states.actions)
        self.walkers.reset(env_states=env_sates, model_states=model_states)
        self.walkers.update_ids()
        if self._use_tree:
            self.tree.reset(
                env_states=self.walkers.env_states,
                model_states=self.walkers.model_states,
                walkers_states=walkers_states,
            )
            self.update_tree([0] * self.walkers.n)

    #@profile
    def run_swarm(
        self,
        model_states: States = None,
        env_states: States = None,
        walkers_states: StatesWalkers = None,
        print_every: int = 1e100,
    ):
        """
        Run a new search process.

        Args:
            model_states: States that define the initial state of the environment.
            env_states: States that define the initial state of the model.
            walkers_states: States that define the internal states of the walkers.
            print_every: Display the algorithm progress every `print_every` epochs.
        Returns:
            None.

        """
        self.reset(model_states=model_states, env_states=env_states)
        self.epoch = 0
        while not self.calculate_end_condition():
            try:
                self.run_step()
                if self.epoch % print_every == 0:
                    print(self)
                    clear_output(True)
                self.epoch += 1
            except KeyboardInterrupt as e:
                break

    def calculate_end_condition(self) -> bool:
        return self.walkers.calculate_end_condition()

    #@profile
    def run_step(self):
        self.walkers.fix_best()
        self.step_walkers()
        old_ids = set(self.walkers.states.id_walkers.copy())
        self.walkers.balance()
        new_ids = set(self.walkers.states.id_walkers)
        self.prune_tree(old_ids=set(old_ids), new_ids=set(new_ids))

    #@profile
    def step_walkers(self):
        """
        Make the walkers undergo a random perturbation process in the swarm \
        Environment.
        """
        model_states = self.walkers.model_states
        env_states = self.walkers.env_states

        states_ids = (
            copy.deepcopy(self.walkers.states.id_walkers).astype(int).flatten().tolist()
            if self._use_tree
            else None
        )

        model_states = self.model.predict(env_states=env_states, model_states=model_states,
                                          walkers_states=self.walkers.states)
        env_states = self.env.step(model_states=model_states, env_states=env_states)
        self.walkers.update_states(
            env_states=env_states, model_states=model_states, end_condition=env_states.ends
        )
        self.walkers.update_ids()
        self.update_tree(states_ids)

    def update_tree(self, states_ids: List[int]):
        """
        Update the states tracked by the tree.

        Args:
            states_ids: list containing the ids of the new states added.

        Returns:
            None.

        """
        if self._use_tree:
            self.tree.add_states(
                parent_ids=states_ids,
                env_states=self.walkers.env_states,
                model_states=self.walkers.model_states,
                walkers_states=self.walkers.states,
            )

    def prune_tree(self, old_ids, new_ids):
        """
        Remove all the branches that are do not have alive walkers at their leaf nodes.

        Args:
            old_ids: ids of the states that were leaf nodes before the cloning process.
            new_ids: ids of the new leaf nodes.

        Returns:
            None.

        """
        if self._prune_tree and self._use_tree:
            self.tree.prune_tree(alive_leafs=new_ids, from_hash=True)


class NoBalance(Swarm):

    def run_step(self):
        self.walkers.update_best()
        self.walkers.fix_best()
        self.step_walkers()
        self.walkers.n_iters += 1

    def calculate_end_condition(self):
        return self.epoch > self.walkers.max_iters