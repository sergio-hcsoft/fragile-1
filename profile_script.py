from plangym import AtariEnvironment, ParallelEnvironment

from fragile.core.env import DiscreteEnv
from fragile.core.models import RandomDiscrete
from fragile.core.states import States
from fragile.core.swarm import Swarm


if __name__ == "__main__":
    import numpy as np
    import pytest
    import torch
    from plangym import AtariEnvironment, ParallelEnvironment

    from fragile.core.env import DiscreteEnv
    from fragile.core.models import RandomDiscrete
    from fragile.core.states import States
    from fragile.core.swarm import Swarm
    from fragile.core.walkers import Walkers

    env = ParallelEnvironment(
        env_class=AtariEnvironment,
        name="MsPacman-ram-v0",
        clone_seeds=True,
        autoreset=True,
        blocking=False,
    )

    state, obs = env.reset()

    states = [state.copy() for _ in range(10)]
    actions = [env.action_space.sample() for _ in range(10)]

    data = env.step_batch(states=states, actions=actions)
    new_states, observs, rewards, ends, infos = data

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
    from IPython.core.display import clear_output

    _ = swarm.run_swarm(print_every=100)