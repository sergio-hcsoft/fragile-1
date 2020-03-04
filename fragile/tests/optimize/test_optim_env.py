import numpy as np
import pytest

from fragile.core.states import States, StatesEnv
from fragile.optimize.env import Function


@pytest.fixture()
def env() -> Function:
    return Function.from_bounds_params(
        function=lambda x: np.ones(len(x)),
        shape=(2,),
        low=np.array([-10, -5]),
        high=np.array([10, 5]),
    )


class TestFunction:
    def test_init(self, env):
        pass

    @pytest.mark.parametrize("batch_size", [1, 10])
    def test_reset_batch_size(self, env, batch_size):
        new_states: StatesEnv = env.reset(batch_size=batch_size)
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
        actions = States(actions=np.ones((1, 2)) * 2, batch_size=1, dt=np.ones((1, 2)))
        new_states: StatesEnv = env.step(actions, states)
        assert isinstance(new_states, States)
        assert new_states.rewards[0].item() == 1

    def shapes_are_the_same(self, env, plangym_env):
        plan_states = plangym_env.reset()
        states = env.reset()
        assert states.rewards.shape == plan_states.shape
