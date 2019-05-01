import numpy as np
import pytest

from fragile.core.base_classes import BaseStates
from fragile.core.models import RandomContinous, RandomDiscrete


@pytest.fixture()
def discrete_model() -> RandomDiscrete:
    return RandomDiscrete(10)


@pytest.fixture()
def continous_model() -> RandomContinous:
    return RandomContinous(low=-1, high=1, shape=(3,))


class TestModel:
    def test_calculate_dt(self, discrete_model: RandomDiscrete):
        n_walkers = 7
        env_states = BaseStates(rewards=np.zeros(n_walkers), n_walkers=n_walkers)
        actions, states = discrete_model.reset()
        act_dt, model_states = discrete_model.calculate_dt(states, env_states)
        assert isinstance(model_states, BaseStates)
        assert len(act_dt) == n_walkers

    def test_predict(self, discrete_model):
        n_walkers = 7
        env_states = BaseStates(rewards=np.zeros(n_walkers), n_walkers=n_walkers)
        actions, states = discrete_model.reset()
        actions, model_states = discrete_model.predict(env_states=env_states, model_states=states)
        assert isinstance(model_states, BaseStates)
        assert len(actions) == n_walkers


class TestContinousModel:
    def test_calculate_dt(self, continous_model: RandomContinous):
        n_walkers = 7
        env_states = BaseStates(rewards=np.zeros(n_walkers), n_walkers=n_walkers)
        actions, states = continous_model.reset()
        act_dt, model_states = continous_model.calculate_dt(states, env_states)
        assert isinstance(model_states, BaseStates)
        assert len(act_dt) == n_walkers

    def test_predict(self, continous_model: RandomContinous):
        n_walkers = 7
        env_states = BaseStates(rewards=np.zeros(n_walkers), n_walkers=n_walkers)
        actions, states = continous_model.reset()
        actions, model_states = continous_model.predict(env_states=env_states, model_states=states)
        assert isinstance(model_states, BaseStates)
        assert len(actions) == n_walkers

    def test_sample(self, continous_model: RandomContinous):
        continous_model.seed(160290)
        sampled = continous_model.sample(1)
        test_val = np.array([0.6908575, 0.8783337, 0.59189284])
        assert np.allclose(sampled, test_val), np.allclose(sampled, test_val, sampled, test_val)

        dif_shape = RandomContinous(low=-1, high=1, shape=(3, 9))
        assert dif_shape.sample(batch_size=6).shape == (6, 3, 9)
