import copy
import torch
import numpy as np
from fragile.base_classes import BaseStates

device_states = "cuda" if torch.cuda.is_available() else "cpu"


class States(BaseStates):
    """
    This class is meant to handle several tensors that will contain the data
    associated with the walkers of a Swarm. This means that each tensor will
    have an extra dimension equal to the number of walkers.

    This class behaves as a dictionary of tensors with some extra functionality
    to make easier the process of cloning the along the walkers dimension.

    In order to define the tensors, a state_dict dictionary needs to be specified
    using the following structure::

        state_dict = {name_1: {"sizes": tuple,
                               device=device,
                               dtype=valid_datatype,
                               },
                      }

    Where tuple is a tuple indicating the shape of the desired tensor, that will
     be accessed using the name_1 attribute of the class.

    Args:
        param_dict: Dictionary defining the attributes of the tensors.
        n_walkers: Number items in the first dimension of the data tensors.
        **kwargs: The name-tensor pairs can also be specified as kwargs.

    """

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

    def clone(self, will_clone: torch.Tensor, compas_ix: torch.Tensor):
        will_clone, compas_ix = will_clone.to(self.device), compas_ix.to(self.device)
        for name in self.keys():
            self[name][will_clone] = self[name][compas_ix][will_clone]

    def update(self, other: "BaseStates" = None, **kwargs):
        other = other if other is not None else kwargs
        for name, val in other.items():
            val = torch.from_numpy(val).to(self.device) if isinstance(val, np.ndarray) else val
            self[name] = val

    def get_params_dict(self):
        pass

    def copy(self) -> "States":
        param_dict = {str(name): val.clone() for name, val in self.items()}
        return States(n_walkers=self.n, **param_dict)
