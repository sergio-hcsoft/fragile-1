from typing import Union, Optional

import numpy as np


class Bounds:
    def __init__(
        self,
        high: Union[np.ndarray, float, int] = np.inf,
        low: Union[np.ndarray, float, int] = -np.inf,
        shape: Optional[tuple] = None,
        dtype: type = None,
    ):

        if shape is None and hasattr(high, "shape"):
            shape = high.shape
        elif shape is None and hasattr(low, "shape"):
            shape = low.shape
        self.shape = shape
        if self.shape is None:
            raise TypeError("If shape is None high or low need to have .shape attribute.")
        self.high = high
        self.low = low
        if dtype is not None:
            self.dtype = dtype
        elif hasattr(high, "dtype"):
            self.dtype = high.dtype
        elif hasattr(low, "dtype"):
            self.dtype = low.dtype
        else:
            self.dtype = type(low)

    def __repr__(self):
        return "{} shape {} dtype {} low {} high {}".format(
            self.__class__.__name__, self.dtype, self.shape, self.low, self.high
        )

    @classmethod
    def from_tuples(cls, bounds) -> "Bounds":
        low, high = [], []
        for lo, hi in bounds:
            low.append(lo)
            high.append(hi)
        low, high = np.array(low), np.array(high)
        return Bounds(low=low, high=high)

    @classmethod
    def from_array(cls, x, scale: float = 1.0) -> "Bounds":
        """

        Args:
            x:
            scale:

        Returns:

        """
        scaled = x
        xmin, xmax = scaled.min(axis=0), scaled.max(axis=0)
        xmin_scaled = np.where(xmin < 0, xmin * scale, xmin / scale)
        xmax_scaled = np.where(xmax < 0, xmax / scale, xmax * scale)
        return Bounds(low=xmin_scaled, high=xmax_scaled)

    def clip(self, points):
        return np.clip(points, self.low, self.high)

    def points_in_bounds(self, points: np.ndarray) -> np.ndarray:
        return (self.clip(points) == points).all(axis=1).flatten()

    def safe_margin(self, high, low) -> "Bounds":
        xmin, xmax = self.low, self.high
        xmin_scaled = np.where(xmin < 0, xmin * low, xmin / low)
        xmax_scaled = np.where(xmax < 0, xmax / high, xmax * high)
        return Bounds(low=xmin_scaled, high=xmax_scaled)

    def to_lims(self):
        xlim, ylim = (self.low[0], self.high[0]), (self.low[1], self.high[1])
        return xlim, ylim
