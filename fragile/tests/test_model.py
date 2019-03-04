import pytest
import numpy as np
from fragile.states import States
from fragile.models import RandomDiscrete


@pytest.fixture()
def model() -> RandomDiscrete:
    return RandomDiscrete(10)


class TestModel:
    def test_calculate_dt(self, model: RandomDiscrete):
        n_walkers = 7
        env_states = States(rewards=np.zeros(n_walkers), n_walkers=n_walkers)
        actions, states = model.reset()
        act_dt, model_states = model.calculate_dt(states, env_states)
        assert isinstance(model_states, States)
        assert len(act_dt) == n_walkers

    def test_predict(self, model):
        n_walkers = 7
        env_states = States(rewards=np.zeros(n_walkers), n_walkers=n_walkers)
        actions, states = model.reset()
        actions, model_states = model.predict(env_states=env_states, model_states=states)
        assert isinstance(model_states, States)
        assert len(actions) == n_walkers
