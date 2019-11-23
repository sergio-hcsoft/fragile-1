import copy
from typing import Generator, List, Tuple, Union

import numpy as np

from fragile.core.utils import hash_numpy


class States:
    """
    Handle several tensors that will contain the data associated with the walkers \
    of a Swarm.

    This means that each tensor will have an extra dimension equal to \
    the number of walkers.

    This class behaves as a dictionary of tensors with some extra functionality
    to make easier the process of cloning the along the walkers dimension.

    In order to define the tensors, a state_dict dictionary needs to be specified
    using the following structure::

        state_dict = {"name_1": {"size": tuple([1]),
                                 "dtype": np.float32,
                                },
                     }

    Where tuple is a tuple indicating the shape of the desired tensor, that will
    be accessed using the name_1 attribute of the class.


    Args:
        batch_size: The number of items in the first dimension of the tensors.
        state_dict: Dictionary defining the attributes of the tensors.
        **kwargs: Data can be directly specified as keyword arguments.
    """

    def __init__(self, batch_size: int, state_dict=None, **kwargs):
        """
        Initialise a :class:`States`.

        Args:
             batch_size: The number of items in the first dimension of the tensors.
             state_dict: Dictionary defining the attributes of the tensors.
             **kwargs: The name-tensor pairs can also be specified as kwargs.

        """
        attr_dict = (
            self._params_to_arrays(state_dict, batch_size) if state_dict is not None else {}
        )
        attr_dict.update(kwargs)
        self._names = list(attr_dict.keys())
        self._attr_dict = attr_dict
        self.update(**self._attr_dict)
        self._n_walkers = batch_size

    def __len__(self):
        """Length is equal to n_walkers."""
        return self.n

    def __getitem__(self, item: str) -> Union[np.ndarray, List[np.ndarray]]:
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
            except AttributeError as e:
                raise TypeError("Tried to get a non existing attribute with key {}".format(item))
        else:
            raise TypeError("item must be an instance of str, got {} instead".format(item))

    def __setitem__(self, key, value: Union[Tuple, List, np.ndarray]):
        """
        Allow the class to set its attributes as if it was a dict.

        Args:
            key: Attribute to be set.
            value: Value of the target attribute.

        Returns:
            None.

        """
        if key not in self._names:
            self._names.append(key)
        self.update(**{key: value})

    def __repr__(self):
        string = "{} with {} walkers\n".format(self.__class__.__name__, self.n)
        for k, v in self.items():
            shape = v.shape if hasattr(v, "shape") else None
            new_str = "{}: {} {}\n".format(k, type(v), shape)
            string += new_str
        return string

    def __hash__(self) -> int:
        return hash(tuple([hash_numpy(x) if isinstance(x, np.ndarray) else hash(x) for x in
                    self.vals()]))

    def group_hash(self, name: str) -> int:
        val = getattr(self, name)
        return hash_numpy(val) if isinstance(val, np.ndarray) else hash(val)

    def hash_values(self, name: str) -> List[int]:
        values = getattr(self, name)
        return [hash_numpy(val) if isinstance(val, np.ndarray) else hash(val) for val in values]

    @classmethod
    def concat_states(cls, states: List["States"]) -> "States":
        """
        Transform a list containing states with only one walker to a single \
        States instance with many walkers.
        """
        n_walkers = sum([s.n for s in states])
        names = list(states[0].keys())
        state_dict = {}
        for name in names:
            shape = tuple([n_walkers]) + tuple(states[0][name].shape)
            state_dict[name] = np.concatenate(tuple([s[name] for s in states])).reshape(shape)
        s = cls(batch_size=n_walkers, **state_dict)
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
        """Return a generator for the attribute names of the stored data."""
        return (n for n in self._names)

    def vals(self) -> Generator:
        """Return a generator for the values of the stored data."""
        return (self[name] for name in self._names)

    def items(self) -> Generator:
        """Return a generator for the attribute names and the values of the stored data."""
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
            values = [v[i] if isinstance(v, np.ndarray) else v for v in self.vals()]
            yield tuple(self._names), tuple(values)

    def split_states(self) -> "States":
        """
        Return a generator for n_walkers different states, where each one \
        contain only the  data corresponding to one walker.
        """
        for k, v in self.iteritems():
            yield self.__class__(batch_size=1, **dict(zip(k, v)))

    def update(self, other: "States" = None, **kwargs):
        """
        Modify the data stored in the States instance.

        Existing attributes will be updated, and new attributes will be created if needed.

        Args:
            other: State class that will be copied upon update.
            **kwargs: It is possible to specify the update as key value attributes, \
                     where key is the name of the attribute to be updated, and value \
                      is the new value for the attribute.
        """

        def update_or_set_attributes(attrs: Union[dict, States]):
            for name, val in attrs.items():
                try:
                    getattr(self, name)[:] = copy.deepcopy(val)
                except (AttributeError, TypeError, KeyError, ValueError) as e:
                    setattr(self, name, copy.deepcopy(val))

        if other is not None:
            update_or_set_attributes(other)
        if kwargs:
            update_or_set_attributes(kwargs)

    def clone(self, will_clone: np.ndarray, compas_ix: np.ndarray, ignore: set=None):
        """
        Clone all the stored data according to the provided arrays.

        Args:
            will_clone: Array of shape (n_walkers,) of booleans indicating the \
                        index of the walkers that will clone to a random companion.
            compas_ix: Array of integers of shape (n_walkers,). Contains the \
                       indexes of the walkers that will be copied.

        """
        ignore = set() if ignore is None else ignore
        for name in self.keys():
            if isinstance(self[name], np.ndarray) and name not in ignore:
                self[name][will_clone] = self[name][compas_ix][will_clone]

    def get_params_dict(self) -> dict:
        """Return a dictionary containing the data stored in the :class:`States`."""
        return {
            k: {"shape": v.shape, "dtype": v.dtype}
            for k, v in self.__dict__.items()
            if isinstance(v, np.ndarray)
        }

    def copy(self) -> "States":
        """Crete a copy of the current instance."""
        param_dict = {str(name): val.copy() for name, val in self.items()}
        return States(batch_size=self.n, **param_dict)

    @staticmethod
    def _params_to_arrays(param_dict: dict, n_walkers: int) -> dict:
        """
        Create a dictionary containing the arrays specified by param_dict.

        Args:
            param_dict: Dictionary defining the attributes of the tensors.
            n_walkers: Number items in the first dimension of the data tensors.

        Returns:
              Dictionary with the same keys as param_dict, containing arrays specified \
              by `param_dict` values.

        """
        tensor_dict = {}
        for key, val in param_dict.items():
            val_size = val.get("size")
            sizes = n_walkers if val_size is None else tuple([n_walkers]) + val_size
            if "size" in val:
                del val["size"]
            tensor_dict[key] = np.zeros(sizes, **val)
        return tensor_dict
