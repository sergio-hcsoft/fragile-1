Function minimization example
=============================
.. note::
    You can find this documentation as a Jupyter notebook inside the **examples** folder as
    ``02_function_minimization.ipynb``.

There are many problems where we only need to sample a single point
instead of a trajectory. The ``optimize`` module is designed for this
use case. It provide environments and models that help explore function
landscapes in order to find points that meet a desired Min/Max
condition.

Testing a ``FunctionMapper`` on a benchmark function
----------------------------------------------------

The ``FunctionMapper`` is a ``Swarm`` with updated default parameters
for solving minimization problems. It should be used with a
``Function``, which is an ``Environment`` designed to optimize functions
that return an scalar.

In this first example we will be using a benchmarking environment that
represents the
`Eggholder <https://en.wikipedia.org/wiki/Test_functions_for_optimization>`__
function:

.. figure:: images/eggholder.png
   :alt: eggholder

   eggholder

.. code:: ipython3

    from fragile.optimize.swarm import FunctionMapper
    from fragile.optimize.benchmarks import EggHolder

The EggHolder function is defined in the [-512, 512] interval.

.. code:: ipython3

    print(EggHolder(), EggHolder.get_bounds())


.. parsed-literal::

    EggHolder with function eggholder, obs shape (2,), Bounds shape int64 dtype (2,) low [-512 -512] high [512 512]


And its optimum corresponds to the point (512, 404.2319) with a value of
-959.64066271

.. code:: ipython3

    print(EggHolder().best_state, EggHolder.benchmark)


.. parsed-literal::

    [512.     404.2319] -959.64066271


We will be sampling the random perturbations made to the walkers from a
Gaussian distribution

.. code:: ipython3

    from fragile.core import NormalContinuous
    def gaussian_model(env):
        # Gaussian of mean 0 and std of 10, adapted to the environment bounds
        return NormalContinuous(scale=10, loc=0., bounds=env.bounds)

To run the algorithm we need to pass the environment and the model as
parameters to the ``FunctionMapper``.

.. code:: ipython3

    swarm = FunctionMapper(env=EggHolder,
                           model=gaussian_model,
                           n_walkers=100,
                           max_iters=500,
                          )

.. code:: ipython3

    swarm.run_swarm(print_every=50)


.. parsed-literal::

    EggHolder with function eggholder, obs shape (2,),
    
    Best reward found: -958.2408 , efficiency 0.685, Critic: None
    Walkers iteration 451 Dead walkers: 0.00% Cloned: 0.00%
    
    Walkers States: 
    id_walkers shape (100,) Mean: -332284778998036.500, Std: 5362551796820465664.000, Max: 9186431489427723264.000 Min: -9121433302269767680.000
    compas_clone shape (100,) Mean: 49.500, Std: 28.866, Max: 99.000 Min: 0.000
    processed_rewards shape (100,) Mean: 1.068, Std: 0.598, Max: 1.896 Min: 0.087
    virtual_rewards shape (100,) Mean: 1.161, Std: 1.027, Max: 3.252 Min: 0.030
    cum_rewards shape (100,) Mean: -429.340, Std: 344.057, Max: 410.406 Min: -958.241
    distances shape (100,) Mean: 1.068, Std: 0.604, Max: 1.822 Min: 0.213
    clone_probs shape (100,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    will_clone shape (100,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    alive_mask shape (100,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    end_condition shape (100,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    best_reward shape None Mean: nan, Std: nan, Max: nan Min: nan
    best_obs shape (2,) Mean: 457.708, Std: 53.887, Max: 511.595 Min: 403.821
    best_state shape (2,) Mean: 457.000, Std: 54.000, Max: 511.000 Min: 403.000
    critic_score shape (100,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    
    Env States: 
    rewards shape (100,) Mean: -429.027, Std: 343.589, Max: 410.406 Min: -926.930
    ends shape (100,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    
    Model States: 
    actions shape (100, 2) Mean: -0.606, Std: 10.654, Max: 24.927 Min: -36.085
    dt shape (100,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    critic_score shape (100,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    
    


Sampling a function with a local optimizer
------------------------------------------

A simple gaussian perturbation is a very sub-optimal strategy for
sampling new points. It is possible to improve the performance of the
sampling process if we run a local minimization process after each
random perturbation.

This can be done using the ``MinimizerWrapper`` class, that takes in any
instance of a ``Function`` environment, and performs a local minimization
process after each environment step.

The ``MinimizerWrapper`` uses ``scipy.optimize.minimize`` under the
hood, and it can take any parameter that ``scipy.optimize.minimize``
supports.

.. code:: ipython3

    from fragile.optimize.env import MinimizerWrapper
        
    def optimize_eggholder():
        options = {"maxiter": 10}
        return MinimizerWrapper(EggHolder(), options=options)
        
    swarm = FunctionMapper(env=optimize_eggholder,
                           model=gaussian_model,
                           n_walkers=50,
                           max_iters=201,
                          )

.. code:: ipython3

    swarm.run_swarm(print_every=25)


.. parsed-literal::

    EggHolder with function eggholder, obs shape (2,),
    
    Best reward found: -959.6407 , efficiency 0.758, Critic: None
    Walkers iteration 201 Dead walkers: 0.00% Cloned: 0.00%
    
    Walkers States: 
    id_walkers shape (50,) Mean: -1334956730834383104.000, Std: 4920389966770588672.000, Max: 6914237399253209088.000 Min: -8832487513487620096.000
    compas_clone shape (50,) Mean: 24.500, Std: 14.431, Max: 49.000 Min: 0.000
    processed_rewards shape (50,) Mean: 1.040, Std: 0.587, Max: 2.117 Min: 0.116
    virtual_rewards shape (50,) Mean: 1.091, Std: 1.016, Max: 4.061 Min: 0.044
    cum_rewards shape (50,) Mean: -552.718, Std: 197.814, Max: -126.424 Min: -959.641
    distances shape (50,) Mean: 1.006, Std: 0.614, Max: 2.186 Min: 0.228
    clone_probs shape (50,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    will_clone shape (50,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    alive_mask shape (50,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    end_condition shape (50,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    best_reward shape None Mean: nan, Std: nan, Max: nan Min: nan
    best_obs shape (2,) Mean: 458.116, Std: 53.884, Max: 512.000 Min: 404.232
    best_state shape (2,) Mean: 458.000, Std: 54.000, Max: 512.000 Min: 404.000
    critic_score shape (50,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    
    Env States: 
    rewards shape (50,) Mean: -552.718, Std: 197.814, Max: -126.424 Min: -959.641
    ends shape (50,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    
    Model States: 
    actions shape (50, 2) Mean: -1.049, Std: 9.475, Max: 23.702 Min: -29.835
    dt shape (50,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    critic_score shape (50,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    
    


This significantly increases the performance of the algorithm at the
expense of using more computational resources.

Defining a new problem using a ``Function``
-------------------------------------------

It is possible to optimize any python function that returns an scalar
using a ``Function``, as long as two requirements are met:

-  The function needs to work with batches of points stacked across the
   first dimension of a numpy array.

-  It returns a vector of scalars corresponding to the values of each
   point evaluated.

This will allow the ``Function`` to vectorize the calculations on the
batch of walkers.

We will also need to create a ``Bounds`` class that define the function
domain.

In this example we will optimize a four dimensional *styblinski_tang*
function, which all its coordinates defined in the [-5, 5] interval:

.. figure:: images/styblinski_tang.png
   :alt: styblinski_tang

   styblinski_tang

.. code:: ipython3

    from fragile.core import Bounds
    import numpy

.. code:: ipython3

    def styblinski_tang(x: numpy.ndarray) -> numpy.ndarray:
        return numpy.sum(x ** 4 - 16 * x ** 2 + 5 * x, 1) / 2.0
    
    bounds = Bounds(low=-5, high=5, shape=(4,))
    print(bounds)


.. parsed-literal::

    Bounds shape float64 dtype (4,) low [-5. -5. -5. -5.] high [5. 5. 5. 5.]


To define the new environment we only need to pass those two parameters
to a ``Function``

.. code:: ipython3

    from fragile.optimize.env import Function

.. code:: ipython3

    def local_optimize_styblinsky_tang():
        function = Function(function=styblinski_tang, bounds=bounds)
        options = {"maxiter": 5}
        return MinimizerWrapper(function, options=options)
    
    swarm = FunctionMapper(env=local_optimize_styblinsky_tang,
                           model=gaussian_model,
                           n_walkers=50,
                           max_iters=101,
                          )

.. code:: ipython3

    swarm.run_swarm(print_every=25)


.. parsed-literal::

    Function with function styblinski_tang, obs shape (4,),
    
    Best reward found: -156.6647 , efficiency 0.695, Critic: None
    Walkers iteration 101 Dead walkers: 0.00% Cloned: 0.00%
    
    Walkers States: 
    id_walkers shape (50,) Mean: 473754779102187904.000, Std: 5777536638421506048.000, Max: 8800337807658314752.000 Min: -9101554059192675328.000
    compas_clone shape (50,) Mean: 24.500, Std: 14.431, Max: 49.000 Min: 0.000
    processed_rewards shape (50,) Mean: 1.067, Std: 0.546, Max: 1.998 Min: 0.065
    virtual_rewards shape (50,) Mean: 1.062, Std: 0.825, Max: 2.901 Min: 0.046
    cum_rewards shape (50,) Mean: -118.226, Std: 23.576, Max: -53.542 Min: -156.665
    distances shape (50,) Mean: 1.053, Std: 0.576, Max: 1.924 Min: 0.119
    clone_probs shape (50,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    will_clone shape (50,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    alive_mask shape (50,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    end_condition shape (50,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    best_reward shape None Mean: nan, Std: nan, Max: nan Min: nan
    best_obs shape (4,) Mean: -2.903, Std: 0.000, Max: -2.903 Min: -2.904
    best_state shape (4,) Mean: -2.000, Std: 0.000, Max: -2.000 Min: -2.000
    critic_score shape (50,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    
    Env States: 
    rewards shape (50,) Mean: -116.922, Std: 23.214, Max: -53.542 Min: -156.664
    ends shape (50,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    
    Model States: 
    actions shape (50, 4) Mean: -0.072, Std: 4.279, Max: 5.000 Min: -5.000
    dt shape (50,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    critic_score shape (50,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    
    


We can see how the optimization was successful in finding the global
optima of -156.66468

.. code:: ipython3

    swarm.best_found




.. parsed-literal::

    array([-2.9036179, -2.9030898, -2.9035947, -2.9032462], dtype=float32)



.. code:: ipython3

    swarm.best_reward_found




.. parsed-literal::

    -156.66465759277344



Optimizing a function with Evolutionary Strategies
--------------------------------------------------

It is possible to use the ``fragile`` framework to implement
optimization algorithms that do not rely on a cloning process, such as
Evolutionary Strategies.

If the cloning process is not needed the ``NoBalance`` ``Swarm`` is the
recommended choice. It has the same features of a regular ``Swarm``, but
it does not perform the cloning process.

.. code:: ipython3

    from fragile.core.swarm import NoBalance
    from fragile.optimize.models import ESModel

In this example we will be solving a Lennard-Jonnes cluster of 4
particles, which is a 12-dimensional function with a global minima at -6.

.. code:: ipython3

    from fragile.optimize.benchmarks import LennardJones

.. code:: ipython3

    swarm = NoBalance(env=lambda : LennardJones(n_atoms=4),
                      model=lambda env: ESModel(bounds=env.bounds),
                      accumulate_rewards=False,
                      minimize=True,
                      n_walkers=10,
                      max_iters=5000,
                     )

.. code:: ipython3

    swarm.run_swarm(print_every=25)


.. parsed-literal::

    
    Best reward found: -5.9707 , efficiency 0.000, Critic: None
    Walkers iteration 5001 Dead walkers: 50.00% Cloned: 0.00%
    
    Walkers States: 
    id_walkers shape (10,) Mean: -969645893255145472.000, Std: 4148011912550082560.000, Max: 4808734810587725824.000 Min: -7723777933071844352.000
    compas_clone shape (10,) Mean: 4.500, Std: 2.872, Max: 9.000 Min: 0.000
    processed_rewards shape (10,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    virtual_rewards shape (10,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    cum_rewards shape (10,) Mean: 27353.668, Std: 81408.398, Max: 271574.469 Min: -5.971
    distances shape (10,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    clone_probs shape (10,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    will_clone shape (10,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    alive_mask shape (10,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    end_condition shape (10,) Mean: 0.500, Std: 0.500, Max: 1.000 Min: 0.000
    best_reward shape None Mean: nan, Std: nan, Max: nan Min: nan
    best_obs shape (12,) Mean: -0.119, Std: 0.572, Max: 0.877 Min: -1.166
    best_state shape (12,) Mean: -0.083, Std: 0.276, Max: 0.000 Min: -1.000
    critic_score shape (10,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    
    Env States: 
    rewards shape (10,) Mean: 27354.051, Std: 81408.273, Max: 271574.469 Min: -2.753
    ends shape (10,) Mean: 0.500, Std: 0.500, Max: 1.000 Min: 0.000
    
    Model States: 
    actions shape (10, 12) Mean: 0.020, Std: 0.379, Max: 1.429 Min: -1.500
    dt shape (10,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    critic_score shape (10,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    
    

