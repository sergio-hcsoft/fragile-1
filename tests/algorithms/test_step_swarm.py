from plangym import ClassicControl
import pytest

from fragile.core import DiscreteEnv, DiscreteUniform
from fragile.algorithms import FollowBestModel, StepToBest, StepSwarm
from fragile.distributed.env import ParallelEnv
from tests.core.test_swarm import TestSwarm


def create_majority_step_swarm():
    swarm = StepSwarm(
        model=lambda x: DiscreteUniform(env=x),
        env=lambda: ParallelEnv(lambda: DiscreteEnv(ClassicControl(name="CartPole-v0"))),
        reward_limit=10,
        n_walkers=100,
        max_epochs=20,
        step_epochs=25,
    )
    return swarm


def create_follow_best_step_swarm():
    swarm = StepSwarm(
        root_model=FollowBestModel,
        model=lambda x: DiscreteUniform(env=x),
        env=lambda: ParallelEnv(lambda: DiscreteEnv(ClassicControl("CartPole-v0"))),
        reward_limit=15,
        n_walkers=100,
        max_epochs=15,
        step_epochs=25,
    )
    return swarm


def create_step_to_best():
    swarm = StepToBest(
        model=lambda x: DiscreteUniform(env=x),
        env=lambda: ParallelEnv(lambda: DiscreteEnv(ClassicControl("CartPole-v0"))),
        reward_limit=16,
        n_walkers=100,
        max_epochs=5,
        step_epochs=25,
    )
    return swarm


swarm_dict = {
    "majority": create_majority_step_swarm,
    "follow_best": create_follow_best_step_swarm,
    "step_to_best": create_step_to_best,
}
swarm_names = list(swarm_dict.keys())
test_scores = {
    "majority": 10,
    "follow_best": 15,
    "step_to_best": 19,
}


@pytest.fixture(params=swarm_names, scope="class")
def swarm(request):
    return swarm_dict.get(request.param)()


@pytest.fixture(params=swarm_names, scope="class")
def swarm_with_score(request):
    swarm = swarm_dict.get(request.param)()
    score = test_scores[request.param]
    return swarm, score
