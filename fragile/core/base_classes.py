from typing import Callable, Generator, List, Tuple, Union

import numpy as np
import torch


class BaseStates:
    """
    Handle several tensors that will contain the data associated with the walkers
    of a Swarm. This means that each tensor will have an extra dimension equal to
    the number of walkers.

    This class behaves as a dictionary of tensors with some extra functionality
    to make easier the process of cloning the along the walkers dimension.

    In order to define the tensors, a state_dict dictionary needs to be specified
    using the following structure::

        state_dict = {"name_1": {"sizes": tuple([1]),
                                 "device": "cuda",
                                 "dtype": torch.float32,
                                },
                     }

    Where tuple is a tuple indicating the shape of the desired tensor, that will
    be accessed using the name_1 attribute of the class.


    Args:
        n_walkers: The number of items in the first dimension of the tensors.
        state_dict: Dictionary defining the attributes of the tensors.
        device: Target device where the tensors will be placed.
        **kwargs: The name-tensor pairs can also be specified as kwargs.
    """

    def __init__(self, n_walkers: int, state_dict=None, device=None, **kwargs):

        self.device = device
        attr_dict = (
            self.params_to_tensors(state_dict, n_walkers) if state_dict is not None else kwargs
        )
        self._names = list(attr_dict.keys())
        for key, val in attr_dict.items():
            setattr(self, key, val)
        self._n_walkers = n_walkers

    def __getitem__(
        self, item: Union[str, List[str]]
    ) -> Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]:
        """
        Query an attribute of the class as if it was a dictionary.

        Args:
            item: Name of the attribute to be selected.

        Returns:
            The corresponding item.
        """
        if isinstance(item, str):
            try:
                return getattr(self, item)
            except TypeError as e:
                raise TypeError("Tried to get an attribute with key {}".format(item))
        elif isinstance(item, list):
            return [getattr(self, it) for it in item]
        else:
            raise TypeError(
                "item must be an instance of str or list, got {} instead".format(item)
            )

    def __setitem__(self, key, value: [torch.Tensor, np.ndarray]):
        """
        Allow the class to set its attributes as if it was a dict.

        Args:
            key: Attribute to be set.
            value: Value of the target attribute.

        Returns:
            None
        """
        if isinstance(value, torch.Tensor):
            setattr(self, key, value)
        elif isinstance(value, np.ndarray):
            setattr(self, key, torch.from_numpy(value).to(self.device))
        else:
            raise NotImplementedError(
                "You can only set attributes using torch.Tensors and np.ndarrays"
                "got item value of type {} for key {}".format(type(value), key)
            )

    def __repr__(self):
        string = "{} with {} walkers\n".format(self.__class__.__name__, self.n)
        for k, v in self.items():
            shape = v.shape if hasattr(v, "shape") else None
            new_str = "{}: {} {}\n".format(k, type(v), shape)
            string += new_str
        return string

    @classmethod
    def concat_states(cls, states: List["BaseStates"]) -> "BaseStates":
        """Transform a list containing states with only one walker to a single
         States instance with many walkers.
        """
        n_walkers = sum([s.n for s in states])
        names = list(states[0].keys())
        state_dict = {}
        for name in names:
            shape = tuple([n_walkers]) + tuple(states[0][name].shape)
            state_dict[name] = torch.cat(tuple([s[name] for s in states])).view(shape)
        s = cls(n_walkers=n_walkers, **state_dict)
        return s

    @property
    def n(self) -> int:
        """Return the number of walkers."""
        return self._n_walkers

    def get(self, key: str, default=None):
        """
        Get an attribute by key and return the default value if it does not exist.

        Args:
            key: Attribute to be recovered.
            default: Value returned in case the attribute is not part of state.

        Returns:
            Target attribute if found in the instance, otherwise returns the
             default value.
        """
        if key not in self.keys():
            return default
        return self[key]

    def keys(self) -> Generator:
        return (n for n in self._names)

    def vals(self) -> Generator:
        return (self[name] for name in self._names)

    def items(self) -> Generator:
        return ((name, self[name]) for name in self._names)

    def itervals(self):
        """
        Iterate the states attributes by walker.

        Returns:
            Tuple containing all the names of the attributes, and the values that
            correspond to a given walker.
        """
        if self.n <= 1:
            return self.vals()
        for i in range(self.n):
            yield tuple([v[i] for v in self.vals()])

    def iteritems(self):
        """
        Iterate the states attributes by walker.

        Returns:
            Tuple containing all the names of the attributes, and the values that
            correspond to a given walker.
        """
        if self.n < 1:
            return self.vals()
        for i in range(self.n):
            yield tuple(self._names), tuple([v[i] for v in self.vals()])

    def split_states(self) -> "BaseStates":
        """Generator that returns n_walkers different states, where each one
        contain only the  data corresponding to one walker.
        """
        for k, v in self.iteritems():
            yield self.__class__(n_walkers=1, **dict(zip(k, v)))

    def params_to_tensors(self, param_dict, n_walkers: int):
        """Transforms the param dict into a dict containing the name of the
         attributes as keys, and initialized data structures as values.
        """
        raise NotImplementedError

    def clone(self, will_clone: [np.ndarray, torch.Tensor], compas_ix: [np.ndarray, torch.Tensor]):
        """
        Perform the clone operation on all the data attributes.

        Args:
            will_clone: Array of booleans that will be True when a walker is
             selected to clone. It is a flat array of length equal to n_walkers.
            compas_ix: Array representing the index of the companions selected
             to clone. It is a flat array of length equal to n_walkers.

        Returns:
            None
        """
        raise NotImplementedError

    def update(self, other: "BaseStates" = None, **kwargs):
        """
        Update the data of the internal attributes of the array.

        Args:
            other: State instance that will be used to update the current values.
            **kwargs: It is also possible to update the attributes passing them as
                keyword arguments.

        Returns:
            None
        """
        raise NotImplementedError

    def get_params_dict(self) -> dict:
        """
        Return an state_dict to be used for instantiating an States class.
        In order to define the tensors, a state_dict dictionary needs to be specified
        using the following structure::

            import torch
            state_dict = {"name_1": {"sizes": tuple([1]),
                                     "device": "cuda",
                                     "dtype": torch.float32,
                                   },
                          }

        Where tuple is a tuple indicating the shape of the desired tensor, that will
        be accessed using the name_1 attribute of the class.
        """
        raise NotImplementedError


class BaseEnvironment:
    """
    The Environment is in charge of stepping the walkers, acting as an state
    transition function. For every different problem a new Environment needs to
    be implemented following the BaseEnvironment interface.

    """

    def step(
        self, actions: [np.ndarray, torch.Tensor], env_states: BaseStates, *args, **kwargs
    ) -> BaseStates:
        """
        Step the environment for a batch of walkers.

        Args:
            actions: Batch of actions to be applied to the environment.
            env_states: States representing a batch of states to be set in the
                environment.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            BaseStates representing the next state of the environment and all
            the needed information.
        """
        raise NotImplementedError

    def reset(self, batch_size: int = 1, env_states: BaseStates = None) -> BaseStates:
        """
        Reset the environment and return an States class with batch_size copies
        of the initial state.

        Args:
            batch_size: Number of walkers that the resulting state will have.
            env_states: States class used to set the environment to an arbitrary
             state.

        Returns:
            States class containing the information of the environment after the
             reset.
        """
        raise NotImplementedError

    def get_params_dict(self) -> dict:
        """
        Return an state_dict to be used for instantiating an States class.
        In order to define the tensors, a state_dict dictionary needs to be specified
        using the following structure::

            import torch
            state_dict = {"name_1": {"sizes": tuple([1]),
                                     "device": "cuda",
                                     "dtype": torch.float32,
                                   },
                          }

        Where tuple is a tuple indicating the shape of the desired tensor, that will
        be accessed using the name_1 attribute of the class.
        """
        raise NotImplementedError


class BaseModel:

    """The model is in charge of calculating how the walkers will act with the
    Environment, effectively working as a policy.
    """

    def get_params_dict(self) -> dict:
        """
        Return an state_dict to be used for instantiating an States class.
        In order to define the tensors, a state_dict dictionary needs to be specified
        using the following structure::

            import torch
            state_dict = {"name_1": {"sizes": tuple([1]),
                                     "device": "cuda",
                                     "dtype": torch.float32,
                                   },
                          }

        Where tuple is a tuple indicating the shape of the desired tensor, that will
        be accessed using the name_1 attribute of the class.
        """
        raise NotImplementedError

    def reset(self, batch_size: int = 1) -> BaseStates:
        """
        Restart the model and reset its internal state.

        Args:
            batch_size: Number of elements in the first dimension of the model
            States data.

        Returns:
            BaseStates containing the information of the current
             state of the model (after the reset).
        """
        raise NotImplementedError

    def predict(
        self, model_states: BaseStates, env_states: BaseStates
    ) -> Tuple[Union[np.ndarray, torch.Tensor, BaseStates]]:
        """
        Calculates the next action that needs to be taken at a given state.

        Args:
            model_states: States corresponding to the environment data.
            env_states: States corresponding to the model data.

        Returns:
            tuple(actions, updated model_states)
        """
        raise NotImplementedError

    def calculate_dt(
        self, model_states: BaseStates, env_states: BaseStates
    ) -> Tuple[torch.Tensor, BaseStates]:
        """
        Calculates the number of times that the next action will be applied.

        Args:
           model_states: States corresponding to the environment data.
           env_states: States corresponding to the model data.

        Returns:
           tuple(dt, updated model_states)
        """
        raise NotImplementedError


class BaseWalkers:
    """The Walkers is a data structure that takes care of all the data involved
    in making a Swarm evolve.

    Args:
            n_walkers: Number of walkers the Swarm will contain.
            env_state_params: Contains the structure of the States
                variable with all the information regarding the Environment.
            model_state_params: Contains the structure of the States
                variable with all the information regarding the Model.
            accumulate_rewards: If true accumulate the rewards after each step
                of the environment.
            *args:  Ignored
            **kwargs: Ignored

    """

    def __init__(
        self,
        n_walkers: int,
        env_state_params: dict,
        model_state_params: dict,
        accumulate_rewards: bool = True,
        *args,
        **kwargs
    ):

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
    def env_states(self) -> BaseStates:
        """Return the States class where all the environment information is stored."""
        raise NotImplementedError

    @property
    def model_states(self) -> BaseStates:
        """Return the States class where all the model information is stored."""
        raise NotImplementedError

    def get_env_states(self) -> BaseStates:
        return self.env_states

    def get_model_states(self) -> BaseStates:
        return self.model_states

    def get_observs(self) -> torch.Tensor:
        raise NotImplementedError

    def update_states(self):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        """Restart all the variables needed to perform the fractal evolution process."""
        raise NotImplementedError

    def calc_distances(self):
        """Calculate the distances between the different observations of the walkers."""
        raise NotImplementedError

    def calc_virtual_reward(self):
        """Apply the virtual reward formula to account for all the different goal scores."""
        raise NotImplementedError

    def calc_end_condition(self) -> bool:
        """Return a boolean that controls the stopping of the iteration loop. If True,
        the iteration process stops."""
        raise NotImplementedError

    def update_clone_probs(self, clone_ix, will_clone):
        """Calculate the clone probability of the walkers."""
        raise NotImplementedError

    def update_end_condition(self, ends: [torch.Tensor, np.ndarray]):
        """Update the boundary conditions for the walkers."""
        raise NotImplementedError

    def get_alive_compas(self) -> [torch.Tensor, np.ndarray]:
        """Return an array of indexes corresponding to an alive walker chosen
         at random.
        """
        raise NotImplementedError

    def balance(self):
        raise NotImplementedError


class BaseSwarm:
    """
    The Swarm is in charge of performing a fractal evolution process. It contains
    the necessary logic to use an Environment, a Model, and a Walkers instance
    to run the Swarm evolution algorithm.

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
        self._walkers = None
        self._model = None
        self._env = None
        self.tree = None

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
        """All the policy and random perturbation code (problem specific) will
         be handled here.
         """
        return self._model

    @property
    def walkers(self) -> BaseWalkers:
        return self._walkers

    def init_walkers(self, model_states: "States" = None, env_states: "States" = None):
        """
        Initialize the :class:`fragile.Walkers` and reset their values to start a new search
        process.

        Args:
            model_states: States that define the initial state of the environment.
            env_states: States that define the initial state of the model.

        Returns:

        """
        raise NotImplementedError

    def run_swarm(self, model_states: "States" = None, env_states: "States" = None):
        """
        Run a new search process.
        Args:
            model_states: States that define the initial state of the environment.
            env_states: States that define the initial state of the model.

        Returns:
            None
        """
        raise NotImplementedError

    def step_walkers(self):
        """Make the walkers undergo a random perturbation process in the swarm
        Environment.
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
            *args: Additional args passed to init_swarm.
            **kwargs: Additional kwargs passed to init_swarm.

        Returns:
            None
        """
        raise NotImplementedError
