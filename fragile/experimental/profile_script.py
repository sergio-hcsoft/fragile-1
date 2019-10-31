from plangym import AtariEnvironment, ParallelEnvironment

from fragile.core.env import DiscreteEnv
from fragile.core.models import RandomDiscrete
from fragile.core.swarm import Swarm
from fragile.core.walkers import Walkers


def main():
    env = ParallelEnvironment(
        env_class=AtariEnvironment,
        name="MsPacman-ram-v0",
        clone_seeds=True,
        autoreset=True,
        blocking=False,
    )

    swarm = Swarm(
        model=lambda x: RandomDiscrete(x),
        walkers=Walkers,
        env=lambda: DiscreteEnv(env),
        n_walkers=150,
        skipframe=1,
        max_iters=300000000,
        prune_tree=True,
        reward_scale=2,
    )

    swarm.run_swarm(print_every=100)


if __name__ == "__main__":
    main()
