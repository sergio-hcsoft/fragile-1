import numpy as np
import pytest

from fragile.core.states import States
from fragile.optimize.env import Function


@pytest.fixture()
def env():
    return Function(function=lambda x: np.ones(len(x)), shape=(2,), bounds=[(-10, 10), (-5, 5)])


class TestFunction:
    def test_init(self, env):
        pass

    @pytest.mark.parametrize("batch_size", [1, 10])
    def test_reset_batch_size(self, env, batch_size):
        new_states = env.reset(batch_size=batch_size)
        assert isinstance(new_states, States)
        assert not (new_states.observs == 0).all().item()
        assert (new_states.rewards == 1).all().item(), (
            new_states.rewards,
            new_states.rewards.shape,
        )
        assert (new_states.ends == 0).all().item()
        assert len(new_states.rewards.shape) == 1
        assert new_states.rewards.shape[0] == batch_size
        assert new_states.ends.shape[0] == batch_size
        assert new_states.observs.shape[0] == batch_size
        assert new_states.observs.shape[1] == 2

    def test_step(self, env):
        states = env.reset()
        actions = np.ones((10, 2)) * 2
        dt = np.ones((10, 2))
        new_states = env.step(actions, states)
        assert isinstance(new_states, States)
        assert new_states.rewards[0].item() == 1

    def _test_are_in_bounds(self, env):
        example_1 = np.array([[10, 10], [6, 5], [6, 4], [-11, -6]])
        res_1 = env.are_in_bounds(example_1)
        assert not res_1[0].item()
        assert not res_1[1].item()
        assert res_1[2].item()
        assert not res_1[3].item()

    def shapes_are_the_same(self, env, plangym_env):
        plan_states = plangym_env.reset()
        states = env.reset()
        assert states.rewards.shape == plan_states.shape
