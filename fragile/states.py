import copy
import torch
import numpy as np
from typing import List

device_states = "cuda" if torch.cuda.is_available() else "cpu"


class States:
    """
    This class is meant to handle several tensors that will contain the data
    associated with the walkers of a Swarm. This means that each tensor will
    have an extra dimension equal to the number of walkers.

    This class behaves as a dictionary of tensors with some extra functionality
    to make easier the process of cloning the along the walkers dimension.

    In order to define the tensors, a state_dict dictionary needs to be specified
    using the following structure:

    state_dict = {name_1: {"sizes": tuple,
                           device=device,
                           },
                  }

    Where tuple is a tuple indicating the shape of the desired tensor, that will
    be accessed using the name_1 attribute of the class.


    Args:
        n_walkers:
        state_dict: Dictionary defining the attributes of the tensors.
        **kwargs: The name-tensor pairs can also be specified as kwargs.
    """

    def __init__(self, n_walkers: int, state_dict=None, device=device_states, **kwargs):

        self.device = device
        attr_dict = (
            self.params_to_tensors(state_dict, n_walkers) if state_dict is not None else kwargs
        )
        self._names = list(attr_dict.keys())
        for key, val in attr_dict.items():
            setattr(self, key, val)
        self._n_walkers = n_walkers

    def __getitem__(self, item):
        try:
            return getattr(self, item)
        except TypeError as e:
            raise TypeError(
                "Tried to get an attribute with key {} of type {}".format(item, type(item))
            )

    def __setitem__(self, key, value: [torch.Tensor, np.ndarray]):
        # if isinstance(value, list):
        #    value = np.array(value)
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
    def concat_states(cls, states: List["States"]) -> "States":
        n_walkers = sum([s.n for s in states])
        names = list(states[0].keys())
        state_dict = {}
        for name in names:
            shape = tuple([n_walkers]) + tuple(states[0][name].shape)
            state_dict[name] = torch.cat(tuple([s[name] for s in states])).view(shape)
        s = States(n_walkers=n_walkers, **state_dict)
        return s

    @property
    def n(self):
        return self._n_walkers

    @property
    def state_dict(self):
        return self._state_dict

    def get(self, key):
        return self[key]

    def keys(self):
        return (n for n in self._names)

    def vals(self):
        return (self[name] for name in self._names)

    def items(self):
        return ((name, self[name]) for name in self._names)

    def itervals(self):
        if self.n <= 1:
            return self.vals()
        for i in range(self.n):
            yield tuple([v[i] for v in self.vals()])

    def iteritems(self):
        if self.n < 1:
            return self.vals()
        for i in range(self.n):
            yield tuple(self._names), tuple([v[i] for v in self.vals()])

    def split_states(self) -> "States":
        for k, v in self.iteritems():
            yield States(n_walkers=1, **dict(zip(k, v)))

    def params_to_tensors(self, param_dict, n_walkers: int):
        tensor_dict = {}
        copy_dict = copy.deepcopy(param_dict)
        for key, val in copy_dict.items():
            sizes = tuple([n_walkers]) + val["sizes"]
            del val["sizes"]
            if "device" not in val.keys():
                val["device"] = self.device
            tensor_dict[key] = torch.zeros(sizes, **val)
        return tensor_dict

    def clone(self, will_clone, compas_ix):
        will_clone, compas_ix = will_clone.to(self.device), compas_ix.to(self.device)
        for name in self.keys():
            self[name][will_clone] = self[name][compas_ix][will_clone]

    def update(self, other: "States" = None, **kwargs):
        other = other if other is not None else kwargs
        for name, val in other.items():
            val = torch.from_numpy(val).to(self.device) if isinstance(val, np.ndarray) else val
            self[name] = val
