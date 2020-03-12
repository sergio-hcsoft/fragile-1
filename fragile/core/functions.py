from typing import Callable

import numpy


def relativize(x: numpy.ndarray) -> numpy.ndarray:
    """Normalize the data using a custom smoothing technique."""
    std = x.std()
    if float(std) == 0:
        return numpy.ones(len(x), dtype=type(std))
    standard = (x - x.mean()) / std
    standard[standard > 0] = numpy.log(1.0 + standard[standard > 0]) + 1.0
    standard[standard <= 0] = numpy.exp(standard[standard <= 0])
    return standard


def get_alives_indexes(ends: numpy.ndarray):
    """Get indexes representing random alive walkers given a vector of death conditions."""
    if numpy.all(ends):
        return numpy.arange(len(ends))
    ix = numpy.logical_not(ends).flatten()
    return numpy.random.choice(numpy.arange(len(ix))[ix], size=len(ix), replace=ix.sum() < len(ix))


def calculate_virtual_reward(
    observs: numpy.ndarray,
    rewards: numpy.ndarray,
    ends: numpy.ndarray = None,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    other_reward: numpy.ndarray = 1.0,
    return_compas: bool = False,
    distance_function: Callable = None,
):
    """Calculate the virtual rewards given the required data."""

    def l2_norm(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        return numpy.linalg.norm(x - y, axis=1)

    distance_function = distance_function if distance_function is not None else l2_norm

    compas = get_alives_indexes(ends) if ends is not None else numpy.arange(len(rewards))
    flattened_observs = observs.reshape(len(ends), -1)
    other_reward = (
        other_reward.flatten() if isinstance(other_reward, numpy.ndarray) else other_reward
    )
    distance = distance_function(flattened_observs, flattened_observs[compas])
    distance_norm = relativize(distance.flatten())
    rewards_norm = relativize(rewards)

    virtual_reward = distance_norm ** dist_coef * rewards_norm ** reward_coef * other_reward
    return virtual_reward.flatten() if not return_compas else virtual_reward.flatten(), compas


def calculate_clone(virtual_rewards: numpy.ndarray, ends: numpy.ndarray, eps=1e-3):
    """Calculate the clone indexes and masks from the virtual rewards."""
    compas_ix = get_alives_indexes(ends)
    vir_rew = virtual_rewards.flatten()
    clone_probs = (vir_rew[compas_ix] - vir_rew) / numpy.maximum(vir_rew, eps)
    will_clone = clone_probs.flatten() > numpy.random.random(len(clone_probs))
    return compas_ix, will_clone


def fai_iteration(
    observs: numpy.ndarray,
    rewards: numpy.ndarray,
    ends: numpy.ndarray,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    eps=1e-8,
    other_reward: numpy.ndarray = 1.0,
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
