"""
import numpy as numpy
import pytest

from fragile.optimize.encoder import Critic, Vector


@pytest.fixture()
def encoder():
    return Critic(5, timeout=10)


@pytest.fixture()
def vector():
    return Vector(origin=numpy.array([0, 0]), end=numpy.array([1, 0]), timeout=5)


class TestVector:
    def test_init(self, vector):
        pass

    def test_scalar_product(self, vector):
        other = numpy.array([1, 0])
        res = vector.scalar_product(other)
        assert res == 0
        other = numpy.array([0, 1])
        res = vector.scalar_product(other)
        assert res == 1

    def test_assign_region(self, vector):
        other = numpy.array([1, 0])
        res = vector.assign_region(other)
        assert res == 0
        other = numpy.array([0, 1])
        res = vector.assign_region(other)
        assert res == 1



class TestEncoder:
    def test_init(self, critic):
        pass

    def test_append(self, critic):
        init_len = len(critic)
        start, end = numpy.array([0, 3]), numpy.array([3, 3])
        critic.append(origin=start, end=end, timeout=3)
        assert len(critic) > 0
        assert len(critic) == init_len + 1
        assert isinstance(critic[-1], Vector)

    def test_len(self, critic):
        critic.reset()
        assert len(critic) == 0
        start, end = numpy.array([0, 3]), numpy.array([3, 3])
        for _ in range(15):
            critic.append(origin=start, end=end, timeout=3)

        assert len(critic) == critic.n_vectors, critic
"""
