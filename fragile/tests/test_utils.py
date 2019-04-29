from hypothesis import given
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
import numpy as np


from fragile.utils import calculate_clone_np, calculate_virtual_reward_np, fai_iteration_np


@given(st.integers(), st.integers())
def test_ints_are_commutative(x, y):
    assert x + y == y + x


class TestFaiNumpy:
    @given(
        arrays(np.float32, shape=(10, 3, 10)),
        arrays(np.float32, shape=(10, 1)),
        arrays(np.bool, shape=(10, 1)),
    )
    def test_calculate_clone_np(self, observs, rewards, ends):
        virtual_reward = calculate_virtual_reward_np(observs=observs, rewards=rewards, ends=ends)
        assert isinstance(virtual_reward, np.ndarray)
        assert len(virtual_reward.shape) == 1
        assert len(virtual_reward) == len(rewards)

    @given(arrays(np.float32, shape=13), arrays(np.bool, shape=13), st.floats(1e-7, 1))
    def test_calculate_clone_np(self, virtual_rewards, ends, eps):
        compas_ix, will_clone = calculate_clone_np(
            virtual_rewards=virtual_rewards, ends=ends, eps=eps
        )

        assert isinstance(compas_ix, np.ndarray)
        assert isinstance(will_clone, np.ndarray)

        assert len(compas_ix.shape) == 1
        assert len(will_clone.shape) == 1

        assert len(compas_ix) == len(virtual_rewards)
        assert len(will_clone) == len(virtual_rewards)

        assert isinstance(compas_ix[0], np.int64), type(compas_ix[0])
        assert isinstance(will_clone[0], np.bool_), type(will_clone[0])

    @given(
        arrays(np.float32, shape=(10, 3, 10)),
        arrays(np.float32, shape=(10, 1)),
        arrays(np.bool, shape=(10, 1)),
    )
    def test_fai_iteration_np(self, observs, rewards, ends):
        compas_ix, will_clone = fai_iteration_np(observs=observs, rewards=rewards, ends=ends)
        assert isinstance(compas_ix, np.ndarray)
        assert isinstance(will_clone, np.ndarray)

        assert len(compas_ix.shape) == 1
        assert len(will_clone.shape) == 1

        assert len(compas_ix) == len(rewards)
        assert len(will_clone) == len(rewards)

        assert isinstance(compas_ix[0], np.int64), type(compas_ix[0])
        assert isinstance(will_clone[0], np.bool_), type(will_clone[0])
