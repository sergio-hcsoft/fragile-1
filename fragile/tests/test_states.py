import numpy as np
import pytest
import torch

from fragile.states import States


@pytest.fixture()
def dummy_states():
    n_walkers = 10
    state_dict = {"miau": {"sizes": (1, 100)}, "miau_2": {"sizes": (1, 33)}}
    return States(state_dict=state_dict, n_walkers=n_walkers)


@pytest.fixture()
def states_from_tensor():
    n_walkers = 10
    miau = torch.ones((10, 1, 100))
    miau_2 = torch.zeros((10, 33, 1))
    return States(n_walkers=n_walkers, miau=miau, miau_2=miau_2)


class TestStates:
    def test_init_dict(self, dummy_states):
        pass

    def test_init_kwargs(self, states_from_tensor):
        pass

    def test_access_attr(self, dummy_states):
        assert isinstance(dummy_states.miau, torch.Tensor)
        assert dummy_states.miau.shape[0] == 10
        assert dummy_states.miau_2.shape[2] == 33

    def test_kwargs_access_attr(self, states_from_tensor):
        assert isinstance(states_from_tensor.miau, torch.Tensor)
        assert states_from_tensor.miau.shape[0] == 10
        assert states_from_tensor.miau_2.shape[1] == 33

    def test_key_iter(self, dummy_states):
        names = ["miau", "miau_2"]
        assert all([key == names[i] for i, key in enumerate(dummy_states.keys())])

    def test_get(self, dummy_states):
        assert dummy_states["miau"] is dummy_states.miau

    def test_val_iter(self, dummy_states):
        names = [dummy_states.miau, dummy_states.miau_2]
        assert all([key is names[i] for i, key in enumerate(dummy_states.vals())])

    def test_items_iter(self, dummy_states):
        keys = ["miau", "miau_2"]
        vals = [dummy_states.miau, dummy_states.miau_2]

        assert all(
            [k is keys[i] and v is vals[i] for i, (k, v) in enumerate(dummy_states.items())]
        )

    def test_clone(self, dummy_states):
        dummy_states.miau = torch.arange(dummy_states.n).view(-1, 1)
        dummy_states.miau_2 = torch.arange(dummy_states.n).view(-1, 1)

        will_clone = torch.zeros(dummy_states.n, dtype=torch.uint8)
        will_clone[3:6] = 1
        inverted_range = torch.squeeze(torch.from_numpy(np.array(np.arange(dummy_states.n)[::-1])))
        compas_ix = torch.squeeze(inverted_range)

        dummy_states.clone(will_clone=will_clone, compas_ix=compas_ix)

        target = torch.tensor([[0], [1], [2], [6], [5], [4], [6], [7], [8], [9]])

        assert torch.all(target == dummy_states.miau)
        assert torch.all(target == dummy_states.miau_2)

    def test_update_other(self, dummy_states, states_from_tensor):
        dummy_states.update(other=states_from_tensor)
        for (k1, v1), (k2, v2) in zip(dummy_states.items(), states_from_tensor.items()):
            assert k1 == k2
            assert torch.all(v1 == v2)

    def repr_does_not_crash(self, dummy_states):
        assert isinstance(dummy_states.__repr__(), str)

    def test_setitem_numpy(self, dummy_states):
        dummy_states["miau"] = np.ones(10)
        assert isinstance(dummy_states.miau, torch.Tensor)

    def test_split_states(self, dummy_states):
        for i, state in enumerate(dummy_states.split_states()):
            assert state.n == 1
            for ks, kd in zip(state.keys(), dummy_states.keys()):
                assert ks == kd
                assert torch.all(state[kd] == dummy_states[kd][i])
