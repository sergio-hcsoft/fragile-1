import numpy as np
import pytest  # noqa: F401

from fragile.core.models import RandomContinous, RandomDiscrete
from fragile.core.states import States
from fragile.tests.core.fixtures import continous_model, discrete_model  # noqa: F401


class TestModel:
    def test_calculate_dt(self, discrete_model: RandomDiscrete):
        n_walkers = 7
        states = discrete_model.reset()
        model_states = discrete_model.calculate_dt(batch_size=n_walkers, model_states=states)
        assert isinstance(model_states, States)
        assert len(model_states.dt) == n_walkers

    def test_predict(self, discrete_model):
        n_walkers = 7
        env_states = States(rewards=np.zeros(n_walkers), batch_size=n_walkers)
        states = discrete_model.reset()
        model_states = discrete_model.predict(env_states=env_states, model_states=states)
        assert isinstance(model_states, States)
        assert len(model_states.actions) == n_walkers


class TestContinousModel:
    def test_calculate_dt(self, continous_model: RandomContinous):
        n_walkers = 7
        states = continous_model.reset()
        model_states = continous_model.calculate_dt(batch_size=n_walkers, model_states=states)
        assert isinstance(model_states, States)
        assert len(model_states.dt) == n_walkers

    def test_predict(self, continous_model: RandomContinous):
        n_walkers = 7
        env_states = States(rewards=np.zeros(n_walkers), batch_size=n_walkers)
        states = continous_model.reset()
        model_states = continous_model.predict(env_states=env_states, model_states=states)
        assert isinstance(model_states, States)
        assert len(model_states.actions) == n_walkers

    def test_sample(self, continous_model: RandomContinous):
        continous_model.seed(160290)
        sampled = continous_model.sample(batch_size=1, model_states=continous_model.reset())
        test_val = np.array([-0.18735352, -0.45104676, -0.44690013])
        assert np.allclose(sampled.actions, test_val)
        dif_shape = RandomContinous(low=-1, high=1, shape=(3, 9))
        sampled_shape = dif_shape.sample(
            batch_size=6, model_states=continous_model.reset()
        ).actions.shape
        assert sampled_shape == (6, 3, 9)
