import sys

import pytest

from fragile.distributed.ray.export_swarm import DistributedExport
from tests.core.test_swarm import create_cartpole_swarm


@pytest.fixture()
def ray_swarm(request=None):
    if sys.version_info != (3, 7):
        return DistributedExport(create_cartpole_swarm, n_swarms=2)


class TestDistributedExport:
    @pytest.mark.skipif(sys.version_info > (3, 7), reason="requires python3.7 or lower")
    def test_init(self, ray_swarm):
        pass

    @pytest.mark.skipif(sys.version_info > (3, 7), reason="requires python3.7 or lower")
    def test_reset(self, ray_swarm):
        ray_swarm.reset()

    @pytest.mark.skipif(sys.version_info != (3, 7), reason="requires python3.7 or lower")
    def test_run(self, ray_swarm):
        ray_swarm.run()
        best_walker = ray_swarm.get_best()
        assert best_walker.rewards > 200
