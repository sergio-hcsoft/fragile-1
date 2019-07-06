import copy

import numpy as np

# import torch

# device = "cpu" if not torch.cuda.is_available() else "cuda"


def relativize(x: np.ndarray) -> np.ndarray:
    """Normalize the data using a custom smoothing techinique."""
    std = x.std()
    if float(std) == 0:
        return np.ones(len(x), dtype=type(std))
    standard = (x - x.mean()) / std
    standard[standard > 0] = np.log(1.0 + standard[standard > 0]) + 1.0
    standard[standard <= 0] = np.exp(standard[standard <= 0])
    return standard


def update_defaults(target: dict, **kwargs) -> dict:
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
    return x.mean(), x.std(), x.max(), x.min()


def get_alives_indexes_np(ends: np.ndarray):
    """Get indexes representing random alive walkers given a vector of death conditions."""
    if np.all(ends):
        return np.arange(len(ends))
    ix = np.logical_not(ends).flatten()
    return np.random.choice(np.arange(len(ix))[ix], size=len(ix), replace=ix.sum() < len(ix))


def calculate_virtual_reward(
    observs: np.ndarray,
    rewards: np.ndarray,
    ends: np.ndarray,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    other_reward: np.ndarray = 1.0,
):
    """Calculate the virtual rewards given the required data."""
    compas = get_alives_indexes_np(ends)
    flattened_observs = observs.reshape(len(ends), -1)
    other_reward = other_reward.flatten() if isinstance(other_reward, np.ndarray) else other_reward

    distance = np.linalg.norm(flattened_observs - flattened_observs[compas], axis=1)
    distance_norm = relativize(distance.flatten())
    rewards_norm = relativize(rewards)

    virtual_reward = distance_norm ** dist_coef * rewards_norm ** reward_coef * other_reward
    return virtual_reward.flatten()


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
    virtual_reward = calculate_virtual_reward(
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

def to_numpy(x: [np.ndarray, torch.Tensor, list]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return np.ndarray(x)


def to_tensor(x: [torch.Tensor, np.ndarray, list], device=device, *args, **kwargs) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, list):
        return torch.from_numpy(np.array(x)).to(device)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    else:
        return torch.Tensor(x, device=device, *args, **kwargs)
"""
