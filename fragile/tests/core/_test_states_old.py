import numpy as np
import pytest  # noqa: F401

from fragile.tests.core.fixtures import states, states_from_tensor, walkers_factory  # noqa: F401


class TestStates:
    def test_init_dict(self, states):
        pass

    def test_init_kwargs(self, states_from_tensor):
        pass

    def test_access_attr(self, states):
        assert isinstance(states.miau, np.ndarray)
        assert states.miau.shape[0] == 10
        assert states.miau_2.shape[2] == 33

    def test_kwargs_access_attr(self, states_from_tensor):
        assert isinstance(states_from_tensor.miau, np.ndarray)
        assert states_from_tensor.miau.shape[0] == 10
        assert states_from_tensor.miau_2.shape[1] == 33

    def test_key_iter(self, states):
        names = ["miau", "miau_2"]
        assert all([key == names[i] for i, key in enumerate(states.keys())])

    def test_get(self, states):
        assert states["miau"] is states.miau

    def test_val_iter(self, states):
        names = [states.miau, states.miau_2]
        assert all([key is names[i] for i, key in enumerate(states.vals())])

    def test_items_iter(self, states):
        keys = ["miau", "miau_2"]
        vals = [states.miau, states.miau_2]

        assert all([k is keys[i] and v is vals[i] for i, (k, v) in enumerate(states.items())])

    def test_clone(self, states):
        states.miau = np.arange(states.n).reshape(-1, 1)
        states.miau_2 = np.arange(states.n).reshape(-1, 1)

        will_clone = np.zeros(states.n, dtype=np.bool_)
        will_clone[3:6] = True
        compas_ix = np.array(np.arange(states.n)[::-1])

        states.clone(will_clone=will_clone, compas_ix=compas_ix)

        target = np.array([[0], [1], [2], [6], [5], [4], [6], [7], [8], [9]])

        assert np.all(target == states.miau)
        assert np.all(target == states.miau_2)

    def test_update_other(self, states, states_from_tensor):
        states.update(other=states_from_tensor)
        for (k1, v1), (k2, v2) in zip(states.items(), states_from_tensor.items()):
            assert k1 == k2
            assert np.all(v1 == v2)

    def test_repr_does_not_crash(self, states):
        assert isinstance(states.__repr__(), str)

    def test_setitem_numpy(self, states):
        states["miau"] = np.ones(10)
        assert isinstance(states.miau, np.ndarray)

    def test_split_states(self, states):
        for i, state in enumerate(states.split_states()):
            assert state.n == 1
            for ks, kd in zip(state.keys(), states.keys()):
                assert ks == kd
                assert np.all(state[kd] == states[kd][i])
