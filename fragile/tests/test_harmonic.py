import numpy as np
import pytest

from fragile.core.base_classes import States
from fragile.experimental.harmonic import GausianPerturbator, HarmonicOscillator


@pytest.fixture(scope="module")
def oscillator():
    env = HarmonicOscillator(n_dims=1, k=1, m=1)
    return env


@pytest.fixture()
def model() -> GausianPerturbator:
    return GausianPerturbator(10)


class TestHarmonicOscillator:
    def test_reset(self, oscillator):
        states = oscillator.reset()
        assert isinstance(states, States), states

        batch_size = 10
        states = oscillator.reset(batch_size=batch_size)
        assert isinstance(states, States), states
        assert states.observs.shape[0] == batch_size, states.observs.shape
        assert states.rewards.shape[0] == batch_size

    def test_step(self, oscillator):
        batch_size = 17
        actions = np.zeros((batch_size, 1), dtype=np.float32)
        states = oscillator.reset(batch_size=batch_size)
        new_states = oscillator.step(actions=actions, env_states=states)
        assert new_states.ends.sum() == 0

    def test_physics(self, oscillator, model):
        actions, _ = model.reset()
        states = oscillator.reset(batch_size=10)
        new_states, _, rewards, ends = oscillator._step_harmonic_oscillator(
            actions.reshape(-1, 1), states.observs
        )
        assert states.states.shape == new_states.shape


class TestGausianPerturbator:
    def test_calculate_dt(self, model: GausianPerturbator):
        n_walkers = 7
        env_states = States(rewards=np.zeros((n_walkers, 1)), batch_size=n_walkers)
        actions, states = model.reset()
        act_dt, model_states = model.calculate_dt(states, env_states)
        assert isinstance(model_states, States)
        assert len(act_dt) == n_walkers

    def test_predict(self, model):
        n_walkers = 7
        env_states = States(rewards=np.zeros((n_walkers, 1)), batch_size=n_walkers)
        actions, states = model.reset()
        actions, model_states = model.predict(env_states=env_states, model_states=states)
        assert isinstance(model_states, States)
        assert len(actions) == n_walkers
