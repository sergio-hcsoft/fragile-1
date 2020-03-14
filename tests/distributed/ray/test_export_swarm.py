import sys

import pytest

from fragile.distributed.distributed_export import DistributedExport
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
        reward_limit=51,
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
    "export": 50,
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
class TestExportInterface:
    @pytest.fixture(params=swarm_names, scope="class")
    def swarm_with_score(self, request):
        init_ray()
        swarm = swarm_dict.get(request.param, create_cartpole_swarm)()
        score = test_scores[request.param]
        request.addfinalizer(lambda: kill_swarm(swarm))
        return swarm, score

    @pytest.mark.skipif(True, reason="Still need to fix this")
    def test_score_gets_higher(self, swarm_with_score):
        swarm, target_score = swarm_with_score
        swarm.reset()
        swarm.run()
        reward = swarm.get_best().rewards
        assert reward > target_score, "Iters: {}, rewards: {}".format(
            swarm.walkers.n_iters, swarm.walkers.states.cum_rewards
        )
