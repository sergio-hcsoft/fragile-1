import copy
from typing import Any, Dict, Union

import numpy
from PIL import Image

try:
    from IPython.core.display import clear_output
except ImportError:

    def clear_output(*args, **kwargs):
        """If not using jupyter notebook do nothing."""
        pass


RANDOM_SEED = 160290
random_state = numpy.random.RandomState(seed=RANDOM_SEED)

float_type = numpy.float32
Scalar = Union[int, numpy.int, float, numpy.float]
StateDict = Dict[str, Dict[str, Any]]


def remove_notebook_margin(output_width_pct: int = 80):
    """Make the notebook output wider."""
    from IPython.core.display import HTML

    html = (
        "<style>"
        ".container { width:" + str(output_width_pct) + "% !important; }"
        ".input{ width:70% !important; }"
        ".text_cell{ width:70% !important;"
        " font-size: 16px;}"
        ".title {align:center !important;}"
        "</style>"
    )
    return HTML(html)


def hash_numpy(x: numpy.ndarray) -> int:
    """Return a value that uniquely identifies a numpy array."""
    return hash(x.tostring())


def resize_frame(
    frame: numpy.ndarray, height: int, width: int, mode: str = "RGB"
) -> numpy.ndarray:
    """
    Use PIL to resize an RGB frame to an specified height and width.

    Args:
        frame: Target numpy array representing the image that will be resized.
        height: Height of the resized image.
        width: Width of the resized image.
        mode: Passed to Image.convert.

    Returns:
        The resized frame that matches the provided width and height.

    """
    frame = Image.fromarray(frame)
    frame = frame.convert(mode).resize((height, width))
    return numpy.array(frame)


def update_defaults(target: dict, **kwargs) -> dict:
    """Set the provided data in the target dictionary in case it didn't exist previously."""
    for k, v in kwargs.items():
        target[k] = target.get(k, v)
    return target


def params_to_tensors(param_dict, n_walkers: int):
    """Transform a parameter dictionary into an array dictionary."""
    tensor_dict = {}
    copy_dict = copy.deepcopy(param_dict)
    for key, val in copy_dict.items():
        sizes = tuple([n_walkers]) + val["size"]
        del val["size"]
        tensor_dict[key] = numpy.empty(sizes, **val)
    return tensor_dict


def statistics_from_array(x: numpy.ndarray):
    """Return the (mean, std, max, min) of an array."""
    try:
        return x.mean(), x.std(), x.max(), x.min()
    except AttributeError:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan


"""
def relativize_torch(x, device=device):
    x = x.float()
    std = x.std()
    if float(std) == 0:
        return torch.ones(len(x), device=device, dtype=torch.float32)
    standard = (x - x.mean()) / std
    standard[standard > 0] = torch.log(1.0 + standard[standard > 0]) + 1.0
    standard[standard <= 0] = torch.exp(standard[standard <= 0])
    return standard

def to_numpy(x: [numpy.ndarray, torch.Tensor, list]) -> numpy.ndarray:
    if isinstance(x, numpy.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return numpy.ndarray(x)


def to_tensor(x: [torch.Tensor, numpy.ndarray, list],
              device=device, *args, **kwargs) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, list):
        return torch.from_numpy(numpy.array(x)).to(device)
    elif isinstance(x, numpy.ndarray):
        return torch.from_numpy(x).to(device)
    else:
        return torch.Tensor(x, device=device, *args, **kwargs)
"""
