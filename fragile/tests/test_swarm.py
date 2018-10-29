import torch
from unittest import TestCase
from fragile.swarm import States


class TestInit(TestCase):
    def setUp(self):
        state_dict = {"miau": {"sizes": (1, 100)}}
        self.car = States(n_walkers=10, state_dict=state_dict)

    def test_init(self):
        state_dict = {"miau": {"sizes": (1, 100, 10, 1)}}
        s = States(n_walkers=10, state_dict=state_dict)
        self.assertIsInstance(s.miau, torch.Tensor, "{} is not a Tensor".format(s.miau))
