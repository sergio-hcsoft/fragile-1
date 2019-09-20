from typing import Any, Dict, Optional, Union

import numpy as np

from fragile.core.base_classes import BaseDtSampler
from fragile.core.states import States
from fragile.core.utils import float_type


class GaussianDt(BaseDtSampler):
    """
    Sample an additional vector of clipped gaussian random variables, and \
    stores it in an attribute called `dt`.
    """

    @classmethod
    def get_params_dict(cls) -> Dict[str, Dict[str, Any]]:
        """Return the dictionary with the parameters to create a new `RandomDiscrete` model."""
        params = {"dt": {"dtype": float_type}}
        return params

    def __init__(
        self, min_dt: float = 1., max_dt: float = 1., loc_dt: float = 0.01, scale_dt: float = 1.,
    ):
        """
        Initialize a :class:`DtSampler`.

        Args:
            min_dt: Minimum dt that will be predicted by the model.
            max_dt: Maximum dt that will be predicted by the model.
            loc_dt: Mean of the gaussian random variable that will model dt.
            scale_dt: Standard deviation of the gaussian random variable that will model dt.

        """
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.mean_dt = loc_dt
        self.std_dt = scale_dt

    def calculate_dt(
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
        if batch_size is None and env_states is None:
            raise ValueError("env_states and batch_size cannot be both None.")
        batch_size = batch_size or env_states.n
        dt = self.random_state.normal(loc=self.mean_dt, scale=self.std_dt, size=batch_size)
        dt = np.clip(dt, self.min_dt, self.max_dt)
        return dt
