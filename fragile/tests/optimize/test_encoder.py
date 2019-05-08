import pytest
import torch

from fragile.optimize.encoder import Encoder, Vector


@pytest.fixture()
def encoder():
    return Encoder(5, timeout=10)


@pytest.fixture()
def vector():
    return Vector(origin=torch.tensor([0, 0]), end=torch.tensor([1, 0]), timeout=5)


class TestVector:
    def test_init(self, vector):
        pass

    def test_scalar_product(self, vector):
        other = torch.tensor([1, 0])
        res = vector.scalar_product(other)
        assert res == 0
        other = torch.tensor([0, 1])
        res = vector.scalar_product(other)
        assert res == 1

    def test_assign_region(self, vector):
        other = torch.tensor([1, 0])
        res = vector.assign_region(other)
        assert res == 0
        other = torch.tensor([0, 1])
        res = vector.assign_region(other)
        assert res == 1


class TestEncoder:
    def test_init(self, encoder):
        pass

    def test_append(self, encoder):
        init_len = len(encoder)
        start, end = torch.tensor([0, 3]), torch.tensor([3, 3])
        encoder.append(origin=start, end=end, timeout=3)
        assert len(encoder) > 0
        assert len(encoder) == init_len + 1
        assert isinstance(encoder[-1], Vector)

    def test_len(self, encoder):
        encoder.reset()
        assert len(encoder) == 0
        start, end = torch.tensor([0, 3]), torch.tensor([3, 3])
        for _ in range(15):
            encoder.append(origin=start, end=end, timeout=3)

        assert len(encoder) == encoder.n_vectors, encoder
