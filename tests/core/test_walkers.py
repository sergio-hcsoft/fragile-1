from hypothesis import given
from hypothesis.extra.numpy import arrays
import numpy as np
import pytest

from fragile.core.states import StatesEnv, StatesModel, StatesWalkers
from fragile.core.utils import relativize
from fragile.core.walkers import Walkers


@pytest.fixture()
def states_walkers():
    return StatesWalkers(10)


N_WALKERS = 13


def get_walkers_discrete_gym():
    env_params = {
        "states": {"size": (128,), "dtype": np.int64},
        "observs": {"size": (64, 64, 3), "dtype": np.float32},
        "rewards": {"dtype": np.float32},
        "ends": {"dtype": np.bool_},
    }
    model_params = {
        "actions": {"size": (10,), "dtype": np.int64},
        "dt": {"size": None, "dtype": np.float32},
    }
    return Walkers(
        n_walkers=N_WALKERS, env_state_params=env_params, model_state_params=model_params
    )


def get_function_walkers():
    env_params = {
        "states": {"size": (3,), "dtype": np.int64},
        "observs": {"size": (3,), "dtype": np.float32},
        "rewards": {"dtype": np.float32},
        "ends": {"dtype": np.bool_},
    }
    model_params = {
        "actions": {"size": (3,), "dtype": np.int64},
        "dt": {"size": None, "dtype": np.float32},
    }
    return Walkers(
        n_walkers=N_WALKERS,
        env_state_params=env_params,
        model_state_params=model_params,
        minimize=True,
    )


walkers_config = {"discrete-gym": get_walkers_discrete_gym, "function": get_function_walkers}


@pytest.fixture()
def walkers(request):
    return walkers_config.get(request.param, get_walkers_discrete_gym)()


class TestStatesWalkers:
    def test_reset(self, states_walkers):
        for name in states_walkers.keys():
            assert states_walkers[name] is not None, name
            assert len(states_walkers[name]) == states_walkers.n, name

        states_walkers.reset()
        for name in states_walkers.keys():
            assert states_walkers[name] is not None, name
            assert len(states_walkers[name]) == states_walkers.n, name

    def test_update(self, states_walkers):
        states_walkers = StatesWalkers(10)
        states_walkers.reset()
        test_vals = np.arange(states_walkers.n)
        states_walkers.update(virtual_rewards=test_vals, distances=test_vals)
        assert (states_walkers.virtual_rewards == test_vals).all()
        assert (states_walkers.distances == test_vals).all()


class TestWalkers:
    walkers_fixture_params = ["discrete-gym"]

    @pytest.mark.parametrize("walkers", walkers_fixture_params, indirect=True)
    def test_init(self, walkers):
        pass

    @pytest.mark.parametrize("walkers", walkers_fixture_params, indirect=True)
    def test_repr_not_crashes(self, walkers):
        assert isinstance(walkers.__repr__(), str)

    @pytest.mark.parametrize("walkers", walkers_fixture_params, indirect=True)
    def test_getattr(self, walkers):
        assert isinstance(walkers.states.will_clone, np.ndarray)
        assert isinstance(walkers.env_states.observs, np.ndarray)
        assert isinstance(walkers.env_states, StatesEnv)
        assert isinstance(walkers.model_states, StatesModel)
        with pytest.raises(AttributeError):
            assert isinstance(walkers.moco, np.ndarray)

    @pytest.mark.parametrize("walkers", walkers_fixture_params, indirect=True)
    def test_calculate_end_condition(self, walkers):
        walkers.reset()
        walkers.states.update(end_condition=np.ones(walkers.n))
        assert walkers.calculate_end_condition()
        walkers.states.update(end_condition=np.zeros(walkers.n))
        assert not walkers.calculate_end_condition()
        walkers.max_iters = 10
        walkers.n_iters = 8
        assert not walkers.calculate_end_condition()
        walkers.n_iters = 11
        assert walkers.calculate_end_condition()

    @pytest.mark.parametrize("walkers", walkers_fixture_params, indirect=True)
    def test_alive_compas(self, walkers):
        end_cond = np.ones_like(walkers.states.end_condition).astype(bool).copy()
        end_cond[3] = 0
        walkers.states.end_condition = end_cond
        compas = walkers.get_alive_compas()
        assert np.all(compas == 3), "Type of end_cond: {} end_cond: {}: alive ix: {}".format(
            type(end_cond), end_cond, walkers.states.alive_mask
        )
        assert len(compas.shape) == 1

    @pytest.mark.parametrize("walkers", walkers_fixture_params, indirect=True)
    def test_update_clone_probs(self, walkers):
        walkers.reset()
        walkers.states.update(virtual_rewards=relativize(np.arange(walkers.n)))
        walkers.update_clone_probs()
        assert 0 < np.sum(walkers.states.clone_probs == walkers.states.clone_probs[0]), (
            walkers.states.virtual_rewards,
            walkers.states.clone_probs,
        )
        walkers.reset()
        walkers.update_clone_probs()
        assert np.sum(walkers.states.clone_probs == walkers.states.clone_probs[0]) == walkers.n
        assert walkers.states.clone_probs.shape[0] == walkers.n
        assert len(walkers.states.clone_probs.shape) == 1

    @pytest.mark.parametrize("walkers", walkers_fixture_params, indirect=True)
    def test_balance_not_crashes(self, walkers):
        walkers.reset()
        walkers.balance()
        assert walkers.states.will_clone.sum() == 0

    @pytest.mark.parametrize("walkers", walkers_fixture_params, indirect=True)
    def test_accumulate_rewards(self, walkers):
        walkers.reset()
        walkers._accumulate_rewards = True
        walkers.states.update(cum_rewards={None, 3})  # Override array of Floats and set to None
        walkers.states.update(cum_rewards=None)
        rewards = np.arange(len(walkers))
        walkers._accumulate_and_update_rewards(rewards)
        assert (walkers.states.cum_rewards == rewards).all()
        walkers._accumulate_rewards = False
        walkers.states.update(cum_rewards=np.zeros(len(walkers)))
        rewards = np.arange(len(walkers))
        walkers._accumulate_and_update_rewards(rewards)
        assert (walkers.states.cum_rewards == rewards).all()
        walkers._accumulate_rewards = True
        walkers.states.update(cum_rewards=np.ones(len(walkers)))
        rewards = np.arange(len(walkers))
        walkers._accumulate_and_update_rewards(rewards)
        assert (walkers.states.cum_rewards == rewards + 1).all()

    @pytest.mark.parametrize("walkers", walkers_fixture_params, indirect=True)
    @given(observs=arrays(np.float32, shape=(N_WALKERS, 64, 64, 3)))
    def test_distances_not_crashes(self, walkers, observs):
        walkers.env_states.update(observs=observs)
        walkers.calculate_distances()
        assert isinstance(walkers.states.distances[0], np.float32)
        assert len(walkers.states.distances.shape) == 1
        assert walkers.states.distances.shape[0] == walkers.n
