import pytest
import torch
import numpy as np
from fragile.states import States
from fragile.walkers import Walkers
from fragile.utils import relativize


@pytest.fixture(scope="module")
def walkers():
    n_walkers = 10
    env_dict = {
        "env_1": {"sizes": (1, 100)},
        "env_2": {"sizes": (1, 33)},
        "observs": {"sizes": (1, 100)},
    }
    model_dict = {"model_1": {"sizes": (1, 13)}, "model_2": {"sizes": (1, 5)}}

    walkers = Walkers(
        n_walkers=n_walkers, env_state_params=env_dict, model_state_params=model_dict
    )
    return walkers


@pytest.fixture(scope="module")
def walkers_factory():
    def new_walkers():
        n_walkers = 10
        env_dict = {
            "env_1": {"sizes": (1, 100)},
            "env_2": {"sizes": (1, 33)},
            "observs": {"sizes": (1, 100)},
        }
        model_dict = {"model_1": {"sizes": (1, 13)}, "model_2": {"sizes": (1, 5)}}

        walkers = Walkers(
            n_walkers=n_walkers, env_state_params=env_dict, model_state_params=model_dict
        )
        return walkers

    return new_walkers


class TestWalkers:
    def test_init(self, walkers):
        pass

    def test_repr_not_crashes(self, walkers):
        assert isinstance(walkers.__repr__(), str)

    def test_states_attributes(self, walkers):
        assert isinstance(walkers.env_states, States)
        assert isinstance(walkers.model_states, States)

    def test_getattr(self, walkers):
        assert isinstance(walkers.env_1, torch.Tensor)
        assert isinstance(walkers.model_1, torch.Tensor)
        assert isinstance(walkers.will_clone, torch.Tensor)
        assert isinstance(walkers.observs, torch.Tensor)
        with pytest.raises(AttributeError):
            assert isinstance(walkers.moco, torch.Tensor)

    def test_obs(self, walkers_factory):
        walkers = walkers_factory()
        assert isinstance(walkers.observs, torch.Tensor)
        walkers._env_states.observs = 10
        with pytest.raises(TypeError):
            walkers.observs

        n_walkers = 10
        env_dict = {"env_1": {"sizes": (1, 100)}, "env_2": {"sizes": (1, 33)}}
        model_dict = {"model_1": {"sizes": (1, 13)}, "model_2": {"sizes": (1, 5)}}

        walkers = Walkers(
            n_walkers=n_walkers, env_state_params=env_dict, model_state_params=model_dict
        )
        with pytest.raises(AttributeError):
            walkers.observs

    def test_update_end_condition(self, walkers):
        ends = np.zeros(10)
        walkers.update_end_condition(ends)
        assert torch.all(walkers.end_condition.cpu() == torch.zeros(10, dtype=torch.uint8))
        ends = torch.ones(10, dtype=torch.uint8)
        walkers.update_end_condition(ends)
        assert torch.all(walkers.end_condition.cpu() == ends)

    def test_calculate_end_condition(self, walkers):
        walkers.update_end_condition(np.ones(10))
        assert walkers.calculate_end_cond()
        walkers.update_end_condition(np.zeros(10))
        assert not walkers.calculate_end_cond()

    def test_calculate_distance(self, walkers):
        # TODO: check properly the calculations
        walkers.calc_distances()

    def test_alive_compas(self, walkers):
        end_cond = torch.ones_like(walkers.end_condition)
        end_cond[3] = 0
        walkers.end_condition = end_cond
        compas = walkers.get_alive_compas()
        assert torch.all(compas == 3), "Type of end_cond: {} end_cond: {}: alive ix: {}".format(
            type(end_cond), end_cond, walkers.alive_mask
        )
        assert len(compas.shape) == 1

    def test_update_clone_probs(self, walkers):
        walkers.reset()
        walkers.virtual_rewards[:] = relativize(torch.arange(walkers.n).float().view(-1, 1))
        walkers.update_clone_probs()
        assert 0 < torch.sum(walkers.clone_probs == walkers.clone_probs[0]).cpu().item(), (
            walkers.virtual_rewards,
            walkers.clone_probs,
        )
        walkers.reset()
        walkers.update_clone_probs()
        assert torch.sum(walkers.clone_probs == walkers.clone_probs[0]).item() == walkers.n
        assert walkers.clone_probs.shape[0] == walkers.n
        assert walkers.clone_probs.shape[1] == 1
        assert len(walkers.clone_probs.shape) == 2

    def test_balance(self, walkers_factory):
        walkers = walkers_factory()
        walkers.reset()
        walkers.balance()
        assert walkers.will_clone.sum().cpu().item() == 0

    def test_accumulate_rewards(self, walkers):
        walkers.reset()

    def test_distances(self, walkers):
        walkers.calc_distances()
        assert len(walkers.distances.shape) == 2
        assert walkers.distances.shape[0] == walkers.n
        assert walkers.distances.shape[1] == 1, walkers.distances.shape
