import copy
from typing import Any, Dict, Union

import numpy as np
from PIL import Image

try:
    from IPython.core.display import clear_output
except ImportError:

    def clear_output(*args, **kwargs):
        """If not using jupyter notebook do nothing."""
        pass


RANDOM_SEED = 160290
random_state = np.random.RandomState(seed=RANDOM_SEED)

float_type = np.float32
Scalar = Union[int, np.int, float, np.float]
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


def hash_numpy(x: np.ndarray) -> int:
    """Return a value that uniquely identifies a numpy array."""
    return hash(x.tostring())


def resize_frame(frame: np.ndarray, height: int, width: int, mode: str = "RGB") -> np.ndarray:
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
    return np.array(frame)


def relativize(x: np.ndarray) -> np.ndarray:
    """Normalize the data using a custom smoothing technique."""
    std = x.std()
    if float(std) == 0:
        return np.ones(len(x), dtype=type(std))
    standard = (x - x.mean()) / std
    standard[standard > 0] = np.log(1.0 + standard[standard > 0]) + 1.0
    standard[standard <= 0] = np.exp(standard[standard <= 0])
    return standard


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
        tensor_dict[key] = np.empty(sizes, **val)
    return tensor_dict


def statistics_from_array(x: np.ndarray):
    """Return the (mean, std, max, min) of an array."""
    try:
        return x.mean(), x.std(), x.max(), x.min()
    except AttributeError:
        return np.nan, np.nan, np.nan, np.nan


def get_alives_indexes_np(ends: np.ndarray):
    """Get indexes representing random alive walkers given a vector of death conditions."""
    if np.all(ends):
        return np.arange(len(ends))
    ix = np.logical_not(ends).flatten()
    return np.random.choice(np.arange(len(ix))[ix], size=len(ix), replace=ix.sum() < len(ix))


def calculate_virtual_reward(
    observs: np.ndarray,
    rewards: np.ndarray,
    ends: np.ndarray = None,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    other_reward: np.ndarray = 1.0,
    return_compas: bool = False,
):
    """Calculate the virtual rewards given the required data."""
    compas = get_alives_indexes_np(ends) if ends is not None else np.arange(len(rewards))
    flattened_observs = observs.reshape(len(ends), -1)
    other_reward = other_reward.flatten() if isinstance(other_reward, np.ndarray) else other_reward

    distance = np.linalg.norm(flattened_observs - flattened_observs[compas], axis=1)
    distance_norm = relativize(distance.flatten())
    rewards_norm = relativize(rewards)

    virtual_reward = distance_norm ** dist_coef * rewards_norm ** reward_coef * other_reward
    return virtual_reward.flatten() if not return_compas else virtual_reward.flatten(), compas


def calculate_clone(virtual_rewards: np.ndarray, ends: np.ndarray, eps=1e-3):
    """Calculate the clone indexes and masks from the virtual rewards."""
    compas_ix = get_alives_indexes_np(ends)
    vir_rew = virtual_rewards.flatten()
    clone_probs = (vir_rew[compas_ix] - vir_rew) / np.maximum(vir_rew, eps)
    will_clone = clone_probs.flatten() > np.random.random(len(clone_probs))
    return compas_ix, will_clone


def fai_iteration(
    observs: np.ndarray,
    rewards: np.ndarray,
    ends: np.ndarray,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    eps=1e-8,
    other_reward: np.ndarray = 1.0,
):
    """Perform a FAI iteration."""
    virtual_reward, vr_compas = calculate_virtual_reward(
        observs,
        rewards,
        ends,
        dist_coef=dist_coef,
        reward_coef=reward_coef,
        other_reward=other_reward,
    )
    compas_ix, will_clone = calculate_clone(virtual_rewards=virtual_reward, ends=ends, eps=eps)
    return compas_ix, will_clone


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
