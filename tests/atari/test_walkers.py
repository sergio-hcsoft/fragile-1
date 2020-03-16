import warnings

from hypothesis import given
from hypothesis.errors import HypothesisDeprecationWarning
from hypothesis.extra.numpy import arrays
import numpy
import pytest

from fragile.atari.walkers import AtariWalkers
from tests.core.test_walkers import TestWalkers
from fragile.core.utils import NUMPY_IGNORE_WARNINGS_PARAMS


warnings.filterwarnings("ignore", category=HypothesisDeprecationWarning)

N_WALKERS = 21


def get_atari_walkers_discrete_gym():
    env_params = {
        "states": {"size": (128,), "dtype": numpy.int64},
        "observs": {"size": (160, 210, 3), "dtype": numpy.float32},
        "rewards": {"dtype": numpy.float32},
        "oobs": {"dtype": numpy.bool_},
        "terminals": {"dtype": numpy.bool_},
    }
    model_params = {
        "actions": {"size": (10,), "dtype": numpy.int64},
        "dt": {"size": None, "dtype": numpy.float32},
        "critic": {"size": None, "dtype": numpy.float32},
    }
    return AtariWalkers(
        n_walkers=N_WALKERS, env_state_params=env_params, model_state_params=model_params
    )


walkers_config = {"discrete-atari-gym": get_atari_walkers_discrete_gym}
walkers_fixture_params = ["discrete-atari-gym"]


class TestAtariWalkers(TestWalkers):
    @pytest.fixture(params=walkers_fixture_params, scope="class")
    def walkers(self, request):
        return walkers_config.get(request.param)()

    def test_calculate_end_condition(self, walkers):
        walkers.reset()
        walkers.states.update(oobs=numpy.ones(walkers.n))
        walkers.env_states.update(terminals=numpy.ones(walkers.n))
        assert walkers.calculate_end_condition()

    @given(observs=arrays(numpy.float32, shape=(N_WALKERS, 160, 210, 3)))
    def test_distances_not_crashes(self, walkers, observs):
        with numpy.errstate(**NUMPY_IGNORE_WARNINGS_PARAMS):
            walkers.env_states.update(observs=observs)
            walkers.calculate_distances()
            assert isinstance(walkers.states.distances[0], numpy.float32)
            assert len(walkers.states.distances.shape) == 1
            assert walkers.states.distances.shape[0] == walkers.n
