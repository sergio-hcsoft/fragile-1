import sys

import pytest

from fragile.distributed.distributed_export import DistributedExport
from tests.core.test_swarm import TestSwarm
from tests.distributed.ray import init_ray, ray

using_py38 = sys.version_info >= (3, 8)
using_py36 = sys.version_info < (3, 7)
only_py37 = pytest.mark.skipif(using_py38 or using_py36, reason="requires python3.7")


def create_cartpole_swarm():
    from fragile.core import DiscreteEnv, DiscreteUniform, Swarm
    from plangym.minimal import ClassicControl

    swarm = Swarm(
        model=lambda x: DiscreteUniform(env=x),
        env=lambda: DiscreteEnv(ClassicControl()),
        reward_limit=71,
        n_walkers=50,
        max_iters=100,
        reward_scale=2,
    )
    return swarm


swarm_types = [create_cartpole_swarm]


def create_distributed_export():
    return DistributedExport(create_cartpole_swarm, n_swarms=2)


swarm_dict = {"export": create_distributed_export}
swarm_names = list(swarm_dict.keys())
test_scores = {
    "export": 70,
}


def kill_swarm(swarm):
    try:
        for e in swarm.swarms:
            e.__ray_terminate__.remote()
        swarm.param_server.__ray_terminate__.remote()
    except AttributeError:
        pass
    ray.shutdown()


@only_py37
class TestExportInterface(TestSwarm):
    @pytest.fixture(params=swarm_names, scope="class")
    def swarm(self, request):
        init_ray()
        swarm = swarm_dict.get(request.param, create_cartpole_swarm)()
        request.addfinalizer(lambda: kill_swarm(swarm))
        return swarm

    @pytest.fixture(params=swarm_names, scope="class")
    def swarm_with_score(self, request):
        init_ray()
        swarm = swarm_dict.get(request.param, create_cartpole_swarm)()
        score = test_scores[request.param]
        request.addfinalizer(lambda: kill_swarm(swarm))
        return swarm, score

    # Distributed Swarm does not implement the full interface
    def test_env_init(self, swarm):
        pass

    def test_step_does_not_crashes(self, swarm):
        pass

    def test_attributes(self, swarm):
        pass
