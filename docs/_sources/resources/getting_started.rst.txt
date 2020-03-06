Getting started with Atari games
================================
.. note::
    You can find this documentation as a Jupyter notebook inside the **examples** folder as
    ``01_Introduction_to_fragile_with_Atari_games.ipynb``.


This is a tutorial that explains how to crate a ``Swarm`` to sample
Atari games from the OpenAI ``gym`` library. It covers how to
instantiate a ``Swarm`` and the most important parameters needed to
control the sampling process.

Structure of a ``Swarm``
------------------------

The ``Swarm`` is the class that implements the algorithm’s evolution
loop, and controls all the other classes involved in solving a given
problem:

.. figure:: images/fragile_architecture.png
   :alt: swarm architecture

   swarm architecture

For every problem we want to solve, we will need to define callables
that return instances of the following classes:

-  ``Environment``: Represents problem we want to solve. Given states
   and actions, it returns the next state.
-  ``Model``: It provides an strategy for sampling actions (Policy).
-  ``Walkers``: This class handles the computations of the evolution
   process of the algorithm. The default value should work fine.
-  ``StateTree``: (Optional) it stores the history of states samples by
   the ``Swarm``.
-  ``Critic``: This class implements additional computation, such as a
   new reward, or extra values for our policy.

Choosing to pass Callables to the ``Swarm`` instead of instances is a
desing decission that simplifies the deployment at scale in a cluster,
because it avoids writing tricky serialization code for all the classes.

Defining the ``Environment``
----------------------------

For playing Atari games we will use the interface provided by the
`plangym <https://github.com/Guillemdb/plangym>`__ package. It is a
wraper of OpenAI ``gym`` that allows to easily set and recover the state
of the environments, as well as stepping the environment with batches of
states.

The following code will initialize a ``plangym.Environment`` for an
OpenAI ``gym`` Atari game. The game names use the same convention as the
OpenAI ``gym`` library.

.. code:: ipython3

    from plangym import AtariEnvironment, ParallelEnvironment
    
    game_name = "MsPacman-ram-v0"
    env = ParallelEnvironment(
            env_class=AtariEnvironment,
            name=game_name,
            clone_seeds=True,
            autoreset=True,
            blocking=False,
        )

In order to use a ``plangym.Environment`` in a ``Swarm`` we will need to
define the appropiate Callable object to pass as a parameter.

``fragile`` incorporates a wrapper to use a ``plangym.AtariEnvironment``
that will take care of matching the ``fragile`` API and constructing the
appropiate ``StatesEnv`` class to store its data.

The environment callable does not take any parameters, and must return
an instance of ``fragile.BaseEnvironment``.

.. code:: ipython3

    from fragile.atari.env import AtariEnv
    env_callable = lambda: AtariEnv(env=env)

Defining the ``Model``
----------------------

The ``Model`` defines the policy that will be used to sample the
``Environment``. In this tutorial we will be using a random sapling
strategy over a discrete uniform distribution. This means that every
time we sample an action, the ``Model`` will return an integer in the
range [0, N_actions] for each state.

We will apply each sampled action a given number of time steps. This
number of timesteps will be sampled using the ``GaussianDt``, that is a
``Critic`` that allows to sample a variable number of timesteps for each
action. The number of timesteps will be sampled from a normal
distribution and rounded to an integer.

The model callable passed to the ``Swarm`` takes as a parameter the
``Environment`` and returns an instance of ``Model``.

.. code:: ipython3

    from fragile.core.dt_sampler import GaussianDt
    from fragile.core.models import DiscreteUniform
    dt = GaussianDt(min_dt=3, max_dt=1000, loc_dt=4, scale_dt=2)
    model_callable = lambda env: DiscreteUniform(env=env, critic=dt)

Storing the sampled data inside a ``HistoryTree``
-------------------------------------------------

It is possible to keep track of the sampled data by using a
``HistoryTree``. This data structure will construct a directed acyclic
graph that will contain the sampled states and their transitions.

Passing the ``prune_tree`` parameter to the ``Swarm`` we can choose to
store only the branches of the ``HistoryTree`` that are being explored.
If ``prune_tree`` is ``True`` all the branches of the graph with no
walkers will be removed after every iteration, and if it is ``False``
all the visited states will be kept in memory.

In order to save memory we will be setting it to ``True``.

.. code:: ipython3

    from fragile.core.tree import HistoryTree
    prune_tree = True

Initializing a ``Swarm``
------------------------

Once we have defined the problem-specific callables for the ``Model``
and the ``Environment``, we need to define the parameters used by the
algorithm:

-  ``n_walkers``: This is population size of our algorithm. It defines
   the number of different states that will be explored simultaneously
   at every iteration of the algorithm. It will be equal to the
   ``batch_size`` of the ``States`` (size of the first dimension of the
   data they store).

-  ``max_iters``: Maximum number of iterations that the ``Swarm`` will
   execute. The algorithm will stop either when all the walkers reached
   a death condition, or when the maximum number of iterations is
   reached.

-  ``reward_scale``: Relative importance given to the ``Environment``
   reward with respect to the diversity score of the walkers.

-  ``distance_scale``: Relative importance given to the diversity
   measure of the walkers with respect to their reward.

-  ``minimize``: If ``True``, the ``Swarm`` will try to sample states
   with the lowest reward possible. If ``False`` the ``Swarm`` will
   undergo a maximization process.

.. code:: ipython3

    n_walkers = 64  # A bigger number will increase the quality of the trajectories sampled.
    max_iters = 2000  # Increase to sample longer games.
    reward_scale = 2  # Rewards are more important than diversity.
    distance_scale = 1
    minimize = False  # We want to get the maximum score possible.

.. code:: ipython3

    from fragile.core.swarm import Swarm
    swarm = Swarm(
        model=model_callable,
        env=env_callable,
        tree=HistoryTree,
        n_walkers=n_walkers,
        max_iters=max_iters,
        prune_tree=prune_tree,
        reward_scale=reward_scale,
        distance_scale=distance_scale,
        minimize=minimize,
    )

By printing a ``Swarm`` we can get an overview of the internal data it
contains.

.. code:: ipython3

    print(swarm)

Running the ``Swarm``
---------------------

In order to execute the algorithm we only need to call ``run_swarm``. It
is possible to display the internal data of the ``Swarm`` by using the
``print_every`` parameter. This parameter indicates the number of
iterations that will pass before printing the ``Swarm``.

.. code:: ipython3

    _ = swarm.run_swarm(print_every=50)

Visualizing the sampled game
----------------------------

We will extract the branch of the ``StateTree`` that achieved the
maximum reward and use its states and actions in the
``plangym.Environment``. This way we can render all the trajectory using
the ``render`` provided by the OpenAI gym API.

.. code:: ipython3

    best_ix = swarm.walkers.states.cum_rewards.argmax()
    best_id = swarm.walkers.states.id_walkers[best_ix]
    path = swarm.tree.get_branch(best_id, from_hash=True)
    
    import time
    for s, a in zip(path[0][1:], path[1]):
        env.step(state=s, action=a)
        env.render()
        time.sleep(0.05)
