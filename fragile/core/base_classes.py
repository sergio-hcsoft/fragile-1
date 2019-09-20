from typing import Any, Callable, Dict, Optional

import numpy as np

from fragile.core import RANDOM_SEED, random_state
from fragile.core.states import States


class BaseCritic:

    random_state = random_state

    def calculate(
        self,
        batch_size: int = None,
        model_states: States = None,
        env_states: States = None,
        walkers_states: "StatesWalkers" = None,
    ) -> np.ndarray:
        """
        Calculate the target time step values.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the model data.
            env_states: States corresponding to the environment data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Array containing the target time step.

        """
        raise NotImplementedError

    def reset(self, batch_size: int = 1, model_states: States = None, *args, **kwargs) -> States:
        """
        Restart the DtSampler and reset its internal state.

        Args:
            batch_size: Number of elements in the first dimension of the model \
                        States data.
            model_states: States corresponding to model data. If provided the \
                          model will be reset to this state.
            args: Additional arguments not related to model data.
            kwargs: Additional keyword arguments not related to model data.

        Returns:
            States containing the information of the current state of the \
            model (after the reset).

        """
        pass


class StatesOwner:
    """Every class meant to have its data stored in States must inherit from this class."""

    random_state = random_state
    STATE_CLASS = States

    @classmethod
    def seed(cls, seed: int = RANDOM_SEED):
        """Set the random seed of the random number generator."""
        cls.random_state.seed(seed)

    @classmethod
    def get_params_dict(cls) -> Dict[str, Dict[str, Any]]:
        """
        Return an state_dict to be used for instantiating an States class.

        In order to define the tensors, a state_dict dictionary needs to be specified \
        using the following structure::

            import numpy as np
            state_dict = {"name_1": {"size": tuple([1]),
                                     "dtype": np.float32,
                                   },
                          }

        Where tuple is a tuple indicating the shape of the desired tensor, that \
        will be accessed using the name_1 attribute of the class.
        """
        raise NotImplementedError

    def create_new_states(self, batch_size: int) -> STATE_CLASS:
        """Create new states of given batch_size to store the data of the class."""
        return self.STATE_CLASS(state_dict=self.get_params_dict(), batch_size=batch_size)


class BaseEnvironment(StatesOwner):
    """
    The Environment is in charge of stepping the walkers, acting as an state \
    transition function.

    For every different problem a new Environment needs to be implemented \
    following the :class:`BaseEnvironment` interface.

    """

    def get_params_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Return an state_dict to be used for instantiating the states containing \
        the data describing the Environment.

        In order to define the arrays, a state_dict dictionary needs to be specified \
        using the following structure::

            import numpy as np
            # Example of an state_dict for planning.
            state_dict = {
                "states": {"size": self._env.get_state().shape, "dtype": np.int64},
                "observs": {"size": self._env.observation_space.shape, "dtype": np.float32},
                "rewards": {"dtype": np.float32},
                "ends": {"dtype": np.bool_},
            }

        """
        raise NotImplementedError

    def step(self, model_states: States, env_states: States) -> States:
        """
        Step the environment for a batch of walkers.

        Args:
            model_states: States representing the data to be used to act on the environment..
            env_states: States representing the data to be set in the environment.

        Returns:
            States representing the next state of the environment and all \
            the needed information.

        """
        raise NotImplementedError

    def reset(self, batch_size: int = 1, env_states: States = None, *args, **kwargs) -> States:
        """
        Reset the environment and return an States class with batch_size copies \
        of the initial state.

        Args:
            batch_size: Number of walkers that the resulting state will have.
            env_states: States class used to set the environment to an arbitrary \
                        state.
             args: Additional arguments not related to environment data.
             kwargs: Additional keyword arguments not related to environment data.

        Returns:
            States class containing the information of the environment after the \
             reset.

        """
        raise NotImplementedError


class BaseModel(StatesOwner):
    """
    The model is in charge of calculating how the walkers will act with the \
    Environment, effectively working as a policy.
    """

    def __init__(self, dt_sampler: Optional[BaseCritic] = None):
        """
        Initialize a BaseModel.

        Args:
            dt_sampler: dt_sampler used to calculate an additional time step strategy. \
                        the vector output by this class will multiply the actions of the model.

        """
        self.dt_sampler = dt_sampler

    def get_params_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Return an state_dict to be used for instantiating the states containing \
        the data describing the Model.

        In order to define the arrays, a state_dict dictionary needs to be \
        specified using the following structure::

            import numpy as np
            # Example of an state_dict for a RandomDiscrete Model.
            n_actions = 10
            state_dict = {"actions": {"size": (n_actions,),
                                      "dtype": np.float32,
                                   },
                          "dt": {"size": tuple([n_actions]),
                                 "dtype": np.float32,
                               },
                          }

        Where size is a tuple indicating the shape of the desired tensor, \
        that will be accessed using the actions attribute of the class.
        """
        raise NotImplementedError

    def reset(self, batch_size: int = 1, model_states: States = None, *args, **kwargs) -> States:
        """
        Restart the model and reset its internal state.

        Args:
            batch_size: Number of elements in the first dimension of the model \
                        States data.
            model_states: States corresponding to model data. If provided the \
                          model will be reset to this state.
            args: Additional arguments not related to model data.
            kwargs: Additional keyword arguments not related to model data.

        Returns:
            States containing the information of the current state of the \
            model (after the reset).

        """
        raise NotImplementedError

    def predict(
        self,
        batch_size: int = None,
        model_states: States = None,
        env_states: States = None,
        walkers_states: "StatesWalkers" = None,
    ) -> States:
        """
        Calculate States containing the data needed to interact with the environment.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the model data.
            env_states: States corresponding to the environment data.
            walkers_states: States corresponding to the walkers data.

        Returns:
            Updated model_states with new model data.

        """
        raise NotImplementedError


class BaseWalkers(StatesOwner):
    """
    The Walkers is a data structure that takes care of all the data involved \
    in making a Swarm evolve.
    """

    random_state = random_state

    def __init__(
        self,
        n_walkers: int,
        env_state_params: dict,
        model_state_params: dict,
        accumulate_rewards: bool = True,
    ):
        """
        Initialize a `BaseWalkers`.

        Args:
            n_walkers: Number of walkers the Swarm will contain.
            env_state_params: Contains the structure of the States
                variable with all the information regarding the Environment.
            model_state_params: Contains the structure of the States
                variable with all the information regarding the Model.
            accumulate_rewards: If true accumulate the rewards after each step
                of the environment.

        """
        super(BaseWalkers, self).__init__()
        self.model_state_params = model_state_params
        self.env_state_params = env_state_params
        self.n_walkers = n_walkers
        self.id_walkers = None
        self.death_cond = None
        self._accumulate_rewards = accumulate_rewards

    @property
    def n(self) -> int:
        """Return the number of walkers."""
        return self.n_walkers

    @property
    def env_states(self) -> "States":
        """Return the States class where all the environment information is stored."""
        raise NotImplementedError

    @property
    def model_states(self) -> "States":
        """Return the States class where all the model information is stored."""
        raise NotImplementedError

    @property
    def states(self) -> States:
        """Return the States class where all the model information is stored."""
        raise NotImplementedError

    def get_params_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return the params_dict of the internal StateOwners."""
        state_dict = {
            name: getattr(self, name).get_params_dict()
            for name in {"states", "env_states", "model_states"}
        }
        return state_dict

    def update_states(self, env_states: States = None, model_states: States = None, **kwargs):
        """
        Update the States variables that do not contain internal data and \
        accumulate the rewards in the internal states if applicable.

        Args:
            env_states: States containing the data associated with the Environment.
            model_states: States containing data associated with the Environment.
            **kwargs: Internal states will be updated via keyword arguments.
        """
        raise NotImplementedError

    def reset(
        self,
        model_states: "States" = None,
        env_states: "States" = None,
        walkers_states: "StatesWalkers" = None,
        *args,
        **kwargs
    ):
        """
        Reset a :class:`fragile.Walkers` and clear the internal data to start a \
        new search process.

        Restart all the variables needed to perform the fractal evolution process.

        Args:
            model_states: States that define the initial state of the environment.
            env_states: States that define the initial state of the model.
            walkers_states: States that define the internal states of the walkers.
            args: Additional arguments not related to algorithm data.
            kwargs: Additional keyword arguments not related to algorithm data.

        """
        raise NotImplementedError

    def balance(self):
        """Perform FAI iteration to clone the states."""
        raise NotImplementedError

    def calculate_distances(self):
        """Calculate the distances between the different observations of the walkers."""
        raise NotImplementedError

    def calculate_virtual_reward(self):
        """Apply the virtual reward formula to account for all the different goal scores."""
        raise NotImplementedError

    def calculate_end_condition(self) -> bool:
        """Return a boolean that controls the stopping of the iteration loop. \
        If True, the iteration process stops."""
        raise NotImplementedError

    def clone_walkers(self):
        """Sample the clone probability distribution and clone the walkers accordingly."""
        raise NotImplementedError

    def get_alive_compas(self) -> np.ndarray:
        """
        Return an array of indexes corresponding to an alive walker chosen \
        at random.
        """
        raise NotImplementedError


class BaseSwarm:
    """
    The Swarm is in charge of performing a fractal evolution process.

    It contains the necessary logic to use an Environment, a Model, and a \
    Walkers instance to run the Swarm evolution algorithm.
    """

    def __init__(
        self,
        env: Callable,
        model: Callable,
        walkers: Callable,
        n_walkers: int,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
        *args,
        **kwargs
    ):
        """
        Initialize a :class:`BaseSwarm`.

        Args:
            env: A function that returns an instance of an Environment.
            model: A function that returns an instance of a Model.
            walkers: A callable that returns an instance of BaseWalkers.
            n_walkers: Number of walkers of the swarm.
            reward_scale: Virtual reward exponent for the reward score.
            dist_scale:Virtual reward exponent for the distance score.
            *args: Additional args passed to init_swarm.
            **kwargs: Additional kwargs passed to init_swarm.

        """
        self._walkers = None
        self._model = None
        self._env = None
        self.tree = None
        self.epoch = 0

        self._init_swarm(
            env_callable=env,
            model_callable=model,
            walkers_callable=walkers,
            n_walkers=n_walkers,
            reward_scale=reward_scale,
            dist_scale=dist_scale,
            *args,
            **kwargs
        )

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
    def walkers(self) -> "Walkers":
        """
        Access the :class:`Walkers` in charge of implementing the FAI \
        evolution process.
        """
        return self._walkers

    def reset(
        self,
        model_states: "States" = None,
        env_states: "States" = None,
        walkers_states: "StatesWalkers" = None,
        *args,
        **kwargs
    ):
        """
        Reset a :class:`fragile.Swarm` and clear the isnternal data to start a \
        new search process.

        Args:
            model_states: States that define the initial state of the environment.
            env_states: States that define the initial state of the model.
            walkers_states: States that define the internal states of the walkers.
            args: Additional arguments not related to algorithm data.
            kwargs: Additional keyword arguments not related to algorithm data.
        """
        raise NotImplementedError

    def run_swarm(
        self,
        model_states: "States" = None,
        env_states: "States" = None,
        walkers_states: "StatesWalkers" = None,
    ):
        """
        Run a new search process until the stop condition is met.

        Args:
            model_states: States that define the initial state of the environment.
            env_states: States that define the initial state of the model.
            walkers_states: States that define the internal states of the walkers.

        Returns:
            None.

        """
        raise NotImplementedError

    def step_walkers(self):
        """
        Make the walkers undergo a random perturbation process in the swarm \
        :class:`Environment`.
        """
        raise NotImplementedError

    def _init_swarm(
        self,
        env_callable: Callable,
        model_callable: Callable,
        walkers_callable: Callable,
        n_walkers: int,
        reward_scale: float = 1.0,
        dist_scale: float = 1.0,
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
            dist_scale:Virtual reward exponent for the distance score.

        Returns:
            None.

        """
        raise NotImplementedError
