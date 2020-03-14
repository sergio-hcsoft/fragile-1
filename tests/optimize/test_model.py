import pytest

from fragile.core import Bounds
from fragile.optimize.models import ESModel

from tests.core.test_model import TestModel

BATCH_SIZE = 10


def create_model(name="es_model"):
    if name == "es_model":
        bs = Bounds(low=-10, high=10, shape=(BATCH_SIZE,))
        return lambda: ESModel(bounds=bs)
    raise ValueError("Invalid param `name`.")


model_fixture_params = ["es_model"]


class TestESModel(TestModel):
    BATCH_SIZE = BATCH_SIZE

    @pytest.fixture(scope="class", params=model_fixture_params)
    def model(self, request):
        return create_model(request.param)()

    def test_run_for_1000_predictions(self, model):
        for _ in range(100):
            self.test_predict(model)
