import sys

import pytest

from fragile.distributed.distributed_export import DistributedExport
from tests.core.test_swarm import create_cartpole_swarm, TestSwarm

using_py38 = sys.version_info >= (3, 8)
using_py36 = sys.version_info < (3, 7)
only_py37 = pytest.mark.skipif(using_py38 or using_py36, reason="requires python3.7")

swarm_types = [create_cartpole_swarm]


class TestDistributedExport:
    @pytest.fixture(params=swarm_types)
    def ray_swarm(self, request):
        swarm = DistributedExport(request.param, n_swarms=2)

        def kill_swarm():
            for e in swarm.swarms:
                e.__ray_terminate__.remote()
            swarm.param_server.__ray_terminate__.remote()

        # request.addfinalizer(kill_swarm)
        return swarm

    @pytest.mark.skipif(using_py38, reason="requires python3.7 or lower")
    def test_init(self, ray_swarm):
        pass

    @pytest.mark.skipif(using_py38, reason="requires python3.7 or lower")
    def test_reset(self, ray_swarm):
        ray_swarm.reset()


def create_distributed_export():
    return DistributedExport(create_cartpole_swarm, n_swarms=2)


swarm_dict = {"export": create_distributed_export}
swarm_names = list(swarm_dict.keys())
test_scores = {
    "export": 130,
}


@only_py37
class TestExportInterface(TestSwarm):
    @pytest.fixture(params=swarm_names)
    def swarm(self, request):
        swarm = swarm_dict.get(request.param, create_cartpole_swarm)()
        return swarm

    @pytest.fixture(params=swarm_names)
    def swarm_with_score(self, request):
        swarm = swarm_dict.get(request.param, create_cartpole_swarm)()
        score = test_scores[request.param]

        def kill_swarm():
            for e in swarm.swarms:
                e.__ray_terminate__.remote()
            swarm.param_server.__ray_terminate__.remote()

        request.addfinalizer(kill_swarm)
        return swarm, score

    # Distributed Swarm does not implement the full interface
    def test_env_init(self, swarm):
        pass

    def test_step_does_not_crashes(self, swarm):
        pass

    def test_attributes(self, swarm):
        pass
