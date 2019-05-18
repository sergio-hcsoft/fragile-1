from scipy.optimize import minimize

from fragile.core.utils import to_numpy, to_tensor
from fragile.optimize.env import Function


class Minimizer:
    def __init__(self, function: Function, bounds=None, *args, **kwargs):
        self.env = function
        self.function = function.function
        self.bounds = self.env.bounds if bounds is None else bounds
        self.args = args
        self.kwargs = kwargs

    def minimize(self, x):
        def _optimize(x):
            x = to_tensor(x).view(1, -1)
            y = self.function(x)
            return -float(y)

        num_x = to_numpy(x)
        return minimize(_optimize, num_x, bounds=self.bounds, *self.args, **self.kwargs)
