import warnings

from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.errors import HypothesisDeprecationWarning
import hypothesis.strategies as st
import numpy


from fragile.core.functions import calculate_clone, calculate_virtual_reward, fai_iteration
from fragile.core.utils import NUMPY_IGNORE_WARNINGS_PARAMS

warnings.filterwarnings("ignore", category=HypothesisDeprecationWarning)


@given(st.integers(), st.integers())
def test_ints_are_commutative(x, y):
    assert x + y == y + x


class TestFaiNumpy:
    @given(
        arrays(numpy.float32, shape=(10, 3, 10)),
        arrays(numpy.float32, shape=10),
        arrays(numpy.bool, shape=(10, 1)),
    )
    def test_calculate_reward(self, observs, rewards, oobs):
        with numpy.errstate(**NUMPY_IGNORE_WARNINGS_PARAMS):
            virtual_reward, compas = calculate_virtual_reward(
                observs=observs, rewards=rewards, oobs=oobs
            )
            assert isinstance(virtual_reward, numpy.ndarray)
            assert len(virtual_reward.shape) == 1
            assert len(virtual_reward) == len(rewards)

    @given(arrays(numpy.float32, shape=13), arrays(numpy.bool, shape=13), st.floats(1e-7, 1))
    def test_calculate_clone(self, virtual_rewards, oobs, eps):
        with numpy.errstate(**NUMPY_IGNORE_WARNINGS_PARAMS):
            compas_ix, will_clone = calculate_clone(
                virtual_rewards=virtual_rewards, oobs=oobs, eps=eps
            )

            assert isinstance(compas_ix, numpy.ndarray)
            assert isinstance(will_clone, numpy.ndarray)

            assert len(compas_ix.shape) == 1
            assert len(will_clone.shape) == 1

            assert len(compas_ix) == len(virtual_rewards)
            assert len(will_clone) == len(virtual_rewards)

            assert isinstance(compas_ix[0], numpy.int64), type(compas_ix[0])
            assert isinstance(will_clone[0], numpy.bool_), type(will_clone[0])

    @given(
        arrays(numpy.float32, shape=(10, 3, 10)),
        arrays(numpy.float32, shape=10),
        arrays(numpy.bool, shape=10),
    )
    def test_fai_iteration(self, observs, rewards, oobs):
        with numpy.errstate(**NUMPY_IGNORE_WARNINGS_PARAMS):
            compas_ix, will_clone = fai_iteration(observs=observs, rewards=rewards, oobs=oobs)
            assert isinstance(compas_ix, numpy.ndarray)
            assert isinstance(will_clone, numpy.ndarray)

            assert len(compas_ix.shape) == 1
            assert len(will_clone.shape) == 1

            assert len(compas_ix) == len(rewards)
            assert len(will_clone) == len(rewards)

            assert isinstance(compas_ix[0], numpy.int64), type(compas_ix[0])
            assert isinstance(will_clone[0], numpy.bool_), type(will_clone[0])
