import numpy as np
import pytest

from fragile.core.base_classes import BaseStates
from fragile.core.utils import relativize
from fragile.core.walkers import StatesWalkers, Walkers


@pytest.fixture(scope="module")
def walkers():
    n_walkers = 10
    env_dict = {
        "env_1": {"size": (1, 100)},
        "env_2": {"size": (1, 33)},
        "observs": {"size": (1, 100)},
    }
    model_dict = {"model_1": {"size": (1, 13)}, "model_2": {"size": (1, 5)}}

    walkers = Walkers(
        n_walkers=n_walkers, env_state_params=env_dict, model_state_params=model_dict
    )
    return walkers


@pytest.fixture(scope="module")
def walkers_factory():
    def new_walkers():
        n_walkers = 10
        env_dict = {
            "env_1": {"size": (1, 100)},
            "env_2": {"size": (1, 33)},
            "observs": {"size": (1, 100)},
        }
        model_dict = {"model_1": {"size": (1, 13)}, "model_2": {"size": (1, 5)}}

        walkers = Walkers(
            n_walkers=n_walkers, env_state_params=env_dict, model_state_params=model_dict
        )
        return walkers

    return new_walkers


@pytest.fixture(scope="module")
def states_walkers():
    return StatesWalkers(10)


class TestStatesWalkers:
    def test_reset(self, states_walkers):
        states_walkers.reset()
        for name in states_walkers.keys():
            assert len(states_walkers[name]) == states_walkers.n

    def test_update(self, states_walkers):
        states_walkers.reset()
        test_vals = np.arange(states_walkers.n)
        states_walkers.update(virtual_rewards=test_vals, distances=test_vals)
        assert (states_walkers.virtual_rewards == test_vals).all()
        assert (states_walkers.distances == test_vals).all()


class TestWalkers:
    def test_init(self, walkers):
        pass

    def test_repr_not_crashes(self, walkers):
        assert isinstance(walkers.__repr__(), str)

    def test_states_attributes(self, walkers):
        assert isinstance(walkers.env_states, BaseStates)
        assert isinstance(walkers.model_states, BaseStates)

    def test_getattr(self, walkers):
        assert isinstance(walkers.env_1, np.ndarray)
        assert isinstance(walkers.model_1, np.ndarray)
        assert isinstance(walkers.will_clone, np.ndarray)
        assert isinstance(walkers.observs, np.ndarray)
        with pytest.raises(AttributeError):
            assert isinstance(walkers.moco, np.ndarray)

    def test_obs(self, walkers_factory):
        walkers = walkers_factory()
        assert isinstance(walkers.observs, np.ndarray)
        walkers._env_states.observs = 10

        n_walkers = 10
        env_dict = {"env_1": {"size": (1, 100)}, "env_2": {"size": (1, 33)}}
        model_dict = {"model_1": {"size": (1, 13)}, "model_2": {"size": (1, 5)}}

        walkers = Walkers(
            n_walkers=n_walkers, env_state_params=env_dict, model_state_params=model_dict
        )
        with pytest.raises(AttributeError):
            walkers.observs

    def test_calculate_end_condition(self, walkers):
        walkers.states.update(end_condition=np.ones(10))
        assert walkers.calculate_end_condition()
        walkers.states.update(end_condition=np.zeros(10))
        assert not walkers.calculate_end_condition()

    def test_calculate_distance(self, walkers):
        # TODO: check properly the calculations
        walkers.calculate_distances()

    def test_alive_compas(self, walkers):
        end_cond = np.ones_like(walkers.states.end_condition).astype(bool).copy()
        end_cond[3] = 0
        walkers.states.end_condition = end_cond
        compas = walkers.get_alive_compas()
        assert np.all(compas == 3), "Type of end_cond: {} end_cond: {}: alive ix: {}".format(
            type(end_cond), end_cond, walkers.states.alive_mask
        )
        assert len(compas.shape) == 1

    def test_update_clone_probs(self, walkers):
        walkers.reset()
        walkers.virtual_rewards[:] = relativize(np.arange(walkers.n))
        walkers.update_clone_probs()
        assert 0 < np.sum(walkers.clone_probs == walkers.clone_probs[0]), (
            walkers.virtual_rewards,
            walkers.clone_probs,
        )
        walkers.reset()
        walkers.update_clone_probs()
        assert np.sum(walkers.clone_probs == walkers.clone_probs[0]) == walkers.n
        assert walkers.clone_probs.shape[0] == walkers.n
        assert len(walkers.clone_probs.shape) == 1

    def test_balance(self, walkers_factory):
        walkers = walkers_factory()
        walkers.reset()
        walkers.balance()
        assert walkers.will_clone.sum() == 0

    def test_accumulate_rewards(self, walkers):
        walkers.reset()

    def test_distances(self, walkers):
        walkers.calculate_distances()
        assert len(walkers.distances.shape) == 1
        assert walkers.distances.shape[0] == walkers.n
