.. include:: ../color_roles.rst
Function minimization example
-----------------------------
.. note::
    The notebook version of this example is available in the
    `examples <https://github.com/FragileTech/fragile/tree/master/examples>`_
    section as `02_function_minimization.ipynb <https://github.com/FragileTech/fragile/blob/master/examples/02_function_minimization.ipynb>`_


There are many problems where we only need to sample a single point
instead of a trajectory. The ``optimize`` module is designed for this
use case. It provide environments and models that help explore function
landscapes in order to find points that meet a desired Min/Max
condition.

Testing a ``FunctionMapper`` on a benchmark function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :swarm:`FunctionMapper` is a :class:`Swarm` with updated default parameters
for solving minimization problems. It should be used with a
:env:`Function`, which is an :class:`Environment` designed to optimize functions
that return an scalar.

In this first example we will be using a benchmarking environment that
represents the
`Eggholder <https://en.wikipedia.org/wiki/Test_functions_for_optimization>`__
function:

.. figure:: ../images/02_eggholder.png
   :alt: eggholder

   eggholder

.. code:: ipython3

    from fragile.optimize import FunctionMapper
    from fragile.optimize.benchmarks import EggHolder

The EggHolder function is defined in the [-512, 512] interval.

.. code:: ipython3

    print(EggHolder(), EggHolder.get_bounds())


.. parsed-literal::

    EggHolder with function eggholder, obs shape (2,), Bounds shape int64 dtype (2,) low [-512 -512] high [512 512]


And its optimum corresponds to the point (512, 404.2319) with a value of -959.64066271

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

To run the algorithm we need to pass the environment and the model callables as
parameters to the :swarm:`FunctionMapper`.

.. code:: ipython3

    swarm = FunctionMapper(env=EggHolder,
                           model=gaussian_model,
                           n_walkers=100,
                           max_iters=500,
                          )

.. code:: ipython3

    swarm.run(print_every=50)


.. parsed-literal::

    EggHolder with function eggholder, obs shape (2,),
    
    Best reward found: -894.4974 , efficiency 0.682, Critic: None
    Walkers iteration 451 Out of bounds walkers: 0.00% Cloned: 36.00%
    
    Walkers States: 
    id_walkers shape (100,) Mean: 906495863484378624.000, Std: 4959508738323910656.000, Max: 8971991439156781056.000 Min: -9078798782819690496.000
    compas_clone shape (100,) Mean: 49.600, Std: 28.906, Max: 99.000 Min: 0.000
    processed_rewards shape (100,) Mean: 1.084, Std: 0.489, Max: 1.690 Min: 0.002
    virtual_rewards shape (100,) Mean: 0.972, Std: 0.640, Max: 3.233 Min: 0.003
    cum_rewards shape (100,) Mean: -790.774, Std: 86.739, Max: -475.153 Min: -894.497
    distances shape (100,) Mean: 1.013, Std: 0.587, Max: 2.486 Min: 0.170
    clone_probs shape (100,) Mean: 4.795, Std: 36.629, Max: 366.720 Min: -0.997
    will_clone shape (100,) Mean: 0.360, Std: 0.480, Max: 1.000 Min: 0.000
    in_bounds shape (100,) Mean: 0.980, Std: 0.140, Max: 1.000 Min: 0.000
    best_reward shape None Mean: nan, Std: nan, Max: nan Min: nan
    best_obs shape (2,) Mean: -39.743, Std: 425.954, Max: 386.211 Min: -465.697
    best_state shape (2,) Mean: -39.500, Std: 425.500, Max: 386.000 Min: -465.000
    best_id shape None Mean: nan, Std: nan, Max: nan Min: nan
    
    Env States: 
    states shape (100, 2) Mean: -37.320, Std: 424.444, Max: 424.000 Min: -511.000
    rewards shape (100,) Mean: -789.777, Std: 86.112, Max: -475.153 Min: -893.765
    oobs shape (100,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    terminals shape (100,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    
    Model States: 
    actions shape (100, 2) Mean: 1.976, Std: 9.903, Max: 31.290 Min: -24.825
    dt shape (100,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    critic_score shape (100,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    

Sampling a function with a local optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple gaussian perturbation is a very sub-optimal strategy for
sampling new points. It is possible to improve the performance of the
sampling process if we run a local minimization process after each
random perturbation.

This can be done using the :env:`MinimizerWrapper` class, that takes in any
instance of a :class:`Function` environment, and performs a local
minimization process after each environment step.

The :env:`MinimizerWrapper` uses ``scipy.optimize.minimize`` under the
hood, and it can take any parameter that ``scipy.optimize.minimize``
supports.

.. code:: ipython3

    from fragile.optimize import MinimizerWrapper
        
    def optimize_eggholder():
        options = {"maxiter": 10}
        return MinimizerWrapper(EggHolder(), options=options)
        
    swarm = FunctionMapper(env=optimize_eggholder,
                           model=gaussian_model,
                           n_walkers=50,
                           max_iters=201,
                          )

.. code:: ipython3

    swarm.run(print_every=25)


.. parsed-literal::

    EggHolder with function eggholder, obs shape (2,),
    
    Best reward found: -959.6407 , efficiency 0.929, Critic: None
    Walkers iteration 201 Out of bounds walkers: 0.00% Cloned: 4.00%
    
    Walkers States: 
    id_walkers shape (50,) Mean: 8570365125154154496.000, Std: 0.000, Max: 8570365125154154496.000 Min: 8570365125154154496.000
    compas_clone shape (50,) Mean: 24.500, Std: 14.431, Max: 49.000 Min: 0.000
    processed_rewards shape (50,) Mean: 1.111, Std: 0.159, Max: 1.133 Min: 0.001
    virtual_rewards shape (50,) Mean: 0.950, Std: 0.339, Max: 3.145 Min: 0.003
    cum_rewards shape (50,) Mean: -959.641, Std: 0.000, Max: -959.641 Min: -959.641
    distances shape (50,) Mean: 0.894, Std: 0.384, Max: 2.775 Min: 0.815
    clone_probs shape (50,) Mean: 7.299, Std: 50.998, Max: 364.275 Min: -0.997
    will_clone shape (50,) Mean: 0.040, Std: 0.196, Max: 1.000 Min: 0.000
    in_bounds shape (50,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    best_reward shape None Mean: nan, Std: nan, Max: nan Min: nan
    best_obs shape (2,) Mean: 458.116, Std: 53.884, Max: 512.000 Min: 404.232
    best_state shape (2,) Mean: 458.000, Std: 54.000, Max: 512.000 Min: 404.000
    best_id shape None Mean: nan, Std: nan, Max: nan Min: nan
    
    Env States: 
    states shape (50, 2) Mean: 458.000, Std: 54.000, Max: 512.000 Min: 404.000
    rewards shape (50,) Mean: -959.641, Std: 0.000, Max: -959.641 Min: -959.641
    oobs shape (50,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    terminals shape (50,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    
    Model States: 
    actions shape (50, 2) Mean: -0.089, Std: 9.910, Max: 26.221 Min: -23.486
    dt shape (50,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    critic_score shape (50,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000


This significantly increases the performance of the algorithm at the
expense of using more computational resources.

Defining a new problem using a :class:`Function`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The objective function
~~~~~~~~~~~~~~~~~~~~~~

It is possible to optimize any python function that returns an scalar
using a :env:`Function`, as long as two requirements are met:

-  The function needs to work with batches of points stacked across the
   first dimension of a numpy array.

-  It returns a vector of scalars corresponding to the values of each
   point evaluated.

This will allow the :env:`Function` to vectorize the calculations on the
batch of walkers.

We will also need to create a :class:`Bounds` class that define the function
domain.

In this example we will optimize a four dimensional **styblinski_tang**
function, which all its coordinates defined in the [-5, 5] interval:

.. figure:: ../images/02_styblinski_tang.png
   :alt: styblinski_tang

   styblinski_tang

.. code:: ipython3

    import numpy
    from fragile.core import Bounds

    def styblinski_tang(x: numpy.ndarray) -> numpy.ndarray:
        return numpy.sum(x ** 4 - 16 * x ** 2 + 5 * x, 1) / 2.0

    bounds = Bounds(low=-5, high=5, shape=(4,))
    print(bounds)


.. parsed-literal::

    Bounds shape float64 dtype (4,) low [-5. -5. -5. -5.] high [5. 5. 5. 5.]


Defining arbitrary boundary conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to define any kind of boundary conditions for the
objective function. This can be done by passing a callable object (such
as a function) to the ``custom_domain_check`` parameter.

The ``custom_domain_check`` function has the following signature:

-  It takes a batch of points as input (same as the ``function``
   parameter).
-  It returns an array of booleans with the same length as the input
   array.
-  Each ``True`` value of the returned array indicates that the
   corresponding point is **outside** the function domain.

The ``custom_domain_check`` will only be applied to the points that are
inside the defined :class:`Bounds`.

.. code:: ipython3

    def my_custom_domain_check(x: numpy.ndarray) -> numpy.ndarray:
        return (numpy.sum(x) > 0.0)

To define the new environment we only need to define the appropriate
``env`` callable passing the target ``function``, the :class:`Bounds`, and
optionally a ``custom_domain_check`` to a :env:`Function`.

Then we can use a :swarm:`FunctionMapper` (or any other kind of :class:`Swarm`) to
perform the optimization process.

.. code:: ipython3

    from fragile.optimize import Function

.. code:: ipython3

    def local_optimize_styblinsky_tang():
        function = Function(function=styblinski_tang, bounds=bounds,
                            custom_domain_check=my_custom_domain_check)
        options = {"maxiter": 10}
        return MinimizerWrapper(function, options=options)

    swarm = FunctionMapper(env=local_optimize_styblinsky_tang,
                           model=lambda env: NormalContinuous(scale=1, loc=0.,
                                                              bounds=env.bounds),
                           n_walkers=50,
                           max_iters=101)

Please be aware that if you use a :env:`MinimizerWrapper` with a
:class:`Function` that has a ``custom_domain_check`` defined you can run into
trouble.

This is because the ``scipy.optimize.minimize`` function that is running
under the hood cannot account for arbitrary boundary conditions. This
can lead to the ``minimize`` function returning only local minima that
are outside the defined ``custom_domain_check``.

.. code:: ipython3

    swarm.run(print_every=25)

We can see how the optimization was successful in finding the global optima of -156.66468

.. code:: ipython3

    swarm.best_obs




.. parsed-literal::

    array([-2.9035306, -2.903532 , -2.9035275, -2.903538 ], dtype=float32)



.. code:: ipython3

    swarm.best_reward




.. parsed-literal::

    -156.66465759277344



Optimizing a function with Evolutionary Strategies
--------------------------------------------------

It is possible to use the ``fragile`` framework to implement
optimization algorithms that do not rely on a cloning process, such as
Evolutionary Strategies.

If the cloning process is not needed the :swarm:`NoBalance`` :class:`Swarm` is the
recommended choice. It has the same features of a regular :swarm:`Swarm`, but
it does not perform the cloning process.

.. code:: ipython3

    from fragile.core import NoBalance
    from fragile.optimize import ESModel

In this example we will be solving a Lennard-Jonnes cluster of 4
particles, which is a 12-dimensional function with a global minima at
-6.

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

    swarm.run(print_every=25)


.. parsed-literal::

    
    Best reward found: -4.4128 , efficiency 0.000, Critic: None
    Walkers iteration 5001 Out of bounds walkers: 40.00% Cloned: 0.00%
    
    Walkers States: 
    id_walkers shape (10,) Mean: 414594831663469312.000, Std: 3482706534469331456.000, Max: 7797802107834368000.000 Min: -3246923136020674048.000
    compas_clone shape (10,) Mean: 4.500, Std: 2.872, Max: 9.000 Min: 0.000
    processed_rewards shape (10,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    virtual_rewards shape (10,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    cum_rewards shape (10,) Mean: 1881.650, Std: 5298.438, Max: 17760.336 Min: -4.413
    distances shape (10,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    clone_probs shape (10,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    will_clone shape (10,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    in_bounds shape (10,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    best_reward shape None Mean: nan, Std: nan, Max: nan Min: nan
    best_obs shape (12,) Mean: -0.051, Std: 0.468, Max: 0.978 Min: -0.859
    best_state shape (12,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    best_id shape None Mean: nan, Std: nan, Max: nan Min: nan
    
    Env States: 
    states shape (10, 12) Mean: 0.025, Std: 0.376, Max: 1.000 Min: -2.000
    rewards shape (10,) Mean: 1882.268, Std: 5298.219, Max: 17760.336 Min: -2.230
    oobs shape (10,) Mean: 0.400, Std: 0.490, Max: 1.000 Min: 0.000
    terminals shape (10,) Mean: 0.000, Std: 0.000, Max: 0.000 Min: 0.000
    
    Model States: 
    actions shape (10, 12) Mean: -0.004, Std: 0.581, Max: 1.500 Min: -1.500
    dt shape (10,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    critic_score shape (10,) Mean: 1.000, Std: 0.000, Max: 1.000 Min: 1.000
    
    

