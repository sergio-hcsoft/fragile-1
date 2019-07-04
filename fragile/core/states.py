"""Extend the BaseStates to perform clone operations."""
import numpy as np

from fragile.core.base_classes import BaseStates


class States(BaseStates):
    """
    This class is meant to handle several tensors that will contain the data \
    associated with the walkers of a Swarm.

    This means that each tensor will have an extra dimension equal to the number \
    of walkers.

    This class behaves as a dictionary of tensors with some extra functionality \
    to make easier the process of cloning the along the walkers dimension.

    In order to define the tensors, a state_dict dictionary needs to be specified \
    using the following structure::

        state_dict = {name_1: {"size": tuple,
                               dtype=valid_datatype,
                               },
                      }

    Where tuple is a tuple indicating the shape of the desired tensor, that will \
    be accessed using the name_1 attribute of the class.

    If no `size` parameter is passed, the resulting vector will have lenght equal \
     to `n_walkers`.

    Args:
        param_dict: Dictionary defining the attributes of the tensors.
        n_walkers: Number items in the first dimension of the data tensors.
        **kwargs: The name-tensor pairs can also be specified as kwargs.

    """

    def __init__(self, *args, **kwargs):
        super(States, self).__init__(*args, **kwargs)

    def params_to_arrays(self, param_dict: dict, n_walkers: int) -> dict:
        """
        Creates a dictionary containing the arrays specified by param_dict.

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

    def clone(self, will_clone: np.ndarray, compas_ix: np.ndarray):
        """
        Clone all the stored data according to the provided arrays.

        Args:
            will_clone: Array of shape (n_walkers,) of booleans indicating the \
                        index of the walkers that will clone to a random companion.
            compas_ix: Array of integers of shape (n_walkers,). Contains the \
                       indexes of the walkers that will be copied.
        """
        for name in self.keys():
            self[name][will_clone] = self[name][compas_ix][will_clone]

    def update(self, other: "BaseStates" = None, **kwargs):
        """
        Modify the data stored in the States instance.

        Existing attributes will be updated, and new attributes will be created if needed.

        Args:
            other: State class that will be copied upon update.
            **kwargs: It is possible to specify the update as key value attributes, \
                     where key is the name of the attribute to be updated, and value \
                      is the new value for the attribute.
        """
        other = other if other is not None else kwargs
        for name, val in other.items():
            val = val if isinstance(val, np.ndarray) else np.array(val)
            self[name] = val

    def get_params_dict(self) -> dict:
        pass

    def copy(self) -> "States":
        """Crete a copy of the current instance."""
        param_dict = {str(name): val.copy() for name, val in self.items()}
        return States(batch_size=self.n, **param_dict)
