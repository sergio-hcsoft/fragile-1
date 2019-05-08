import numpy as np
import pytest

from fragile.core.base_classes import BaseStates  # noqa: F401
from fragile.optimize.models import UnitaryContinuous
from fragile.tests.test_model import continous_model, TestContinousModel  # noqa: F401


@pytest.fixture()
def unitary_model() -> UnitaryContinuous:
    return UnitaryContinuous(low=-1, high=1, shape=(3,))


class TestUnitary(TestContinousModel):
    def test_calculate_dt(self, unitary_model: UnitaryContinuous):
        super(TestUnitary, self).test_calculate_dt(continous_model=unitary_model)

    def test_predict(self, unitary_model: UnitaryContinuous):
        super(TestUnitary, self).test_predict(continous_model=unitary_model)

    def test_sample(self, unitary_model: UnitaryContinuous):
        unitary_model.seed(160290)
        sampled = unitary_model.sample(1)
        assert np.allclose(np.linalg.norm(sampled), 1), (np.linalg.norm(sampled), sampled)

        dif_shape = UnitaryContinuous(low=-1, high=1, shape=(3, 9))
        assert dif_shape.sample(batch_size=6).shape == (6, 3, 9)
