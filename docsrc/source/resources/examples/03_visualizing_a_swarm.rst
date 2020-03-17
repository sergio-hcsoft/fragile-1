Visualizing a Swarm
=======================

.. note::
    The notebook version of this example is available in the `examples` as  ``03_visualizing_a_swarm.ipynb``

It is possible to visualize the evolution of an algorithm run using the
``dataviz`` module. This module allows to stream data to ``holoviews``
plots during a run of the algorithm.

This example will cover several classes that allow to plot different
kinds of visualizations. In order to visualize a ``Swarm`` in the
``jupyter notebook`` the first thing we need to do is loading the
``holoviews`` extension for ``bokeh``.

.. code:: ipython3

    import holoviews
    from fragile.core.utils import remove_notebook_margin
    from fragile.dataviz import AtariViz, LandscapeViz, Summary, SwarmViz, SwarmViz1D
    holoviews.extension("bokeh")
    remove_notebook_margin()  # Make the output cell wider

All the visualization classes wrap a :class:`Swarm` to handle all the data
streaming and visualization logic for plotting the :swarm:`Swarm`’s data.

We will start initializing a Swarm like we did in the last tutorial. We
are not focusing on the performance of the sampling, but using the swarm
just to create the visualizations.

.. code:: ipython3

    from fragile.core import NormalContinuous
    from fragile.optimize import FunctionMapper
    from fragile.optimize.benchmarks import EggHolder

    def gaussian_model(env):
        # Gaussian of mean 0 and std of 10, adapted to the environment bounds
        return NormalContinuous(scale=10, loc=0., bounds=env.bounds)
    swarm = FunctionMapper(env=EggHolder,
                           model=gaussian_model,
                           n_walkers=300,
                           max_iters=750,
                           start_same_pos=True,
                          )

Summary visualization
^^^^^^^^^^^^^^^^^^^^^

This is the simplest and fastest visualization, and it includes a table
with information about the current iteration of the :swarm:`Swarm`, the best
score achieved, and the percentages of deaths and clones.

To initialize it you only have to wrap the :class:`Swarm` you want to
visualize.

.. code:: ipython3

    summary = Summary(swarm)

Once the class is initialized, you need to call the ``plot``
function to initialize the plots and create the :class:`holoviews.DynamicMap`
that will plot the data streamed during the algorithm run. The data streaming
will start when ``run`` is called.

.. code:: ipython3

    summary.plot()

.. image::
    ../images/03_summary.gif

.. code:: ipython3

    summary.run()

Histogram visualizations
^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`SwarmViz1d` can be used in any kind of :swarm:`Swarm`, and it allows
to display no only the summary table and the reward evolution curve, but
also histograms for the reward, distance, and virtual reward
distributions of the walkers.

Using the ``stream_interval`` parameter you can choose the number of
iterations that will pass before the data is streamed to the plot. Data
is streamed every 100 iterations by default.

.. code:: ipython3

    swarm_viz_1d = SwarmViz1D(swarm, stream_interval=25)

.. code:: ipython3

    swarm_viz_1d.plot()

.. image::
    ../images/03_1dviz.gif

.. code:: ipython3

    swarm_viz_1d.run()

2D Visualizations
^^^^^^^^^^^^^^^^^

It is also possible to visualize the walkers’ properties using two
dimensional plots. These plots come specially in handy if you are using
two dimensional embeddings of your state space, but the can also be
applied to visualize the first two dimensions of the sampled state space.

The :class:`LandscapeViz` incorporates visualizations of the walkers
distribution, the rewards, the virtual reward and the distance function.
This is done by interpolating the values of the walkers to create a grid,
where the target value will be displayed using a colormap.

.. code:: ipython3

    landscape_viz = LandscapeViz(swarm, stream_interval=25)

.. code:: ipython3

    %%opts QuadMesh {+framewise} Bivariate {+framewise}
    # Opts is necessary to avoid erratic behaviour when creating big DynamicMaps
    landscape_viz.plot()

.. image::
    ../images/03_landscape.gif

.. code:: ipython3

    landscape_viz.run()

Plotting 2D distributions and histograms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`SwarmViz` class incorporated all the distributions presented
above. All the ``dataviz`` classes allow you to select the
visualizations you want to display by passing a list of their names to
the ``display_plots`` parameter.

Passing **“all”** as a parameter will display all the available
visualizations. If you want to find out what are the available
visualizations for a given class you can call the ``PLOT_NAMES``
attribute of the class.

.. code:: ipython3

    SwarmViz.PLOT_NAMES

.. code:: ipython3

    swarm_viz = SwarmViz(swarm, stream_interval=25, display_plots="all")

.. code:: ipython3

    %%opts QuadMesh {+framewise} Bivariate {+framewise}
    swarm_viz.plot()

.. image::
    ../images/03_swarmviz.gif

.. code:: ipython3

    swarm_viz.run()

Visualizing Atari games
^^^^^^^^^^^^^^^^^^^^^^^

The :class:`AtariViz` class includes all the plots that can help visualize
the sampling process of an Atari game. On top of the visualizations
available on the :class:`SwarmViz1d` class, it allows to display the frame of
the best state sampled.

.. code:: ipython3

    from fragile.dataviz.swarm_viz import AtariViz

We will use the game **Qbert** to show how the :class:`AtariViz` works.

.. code:: ipython3

    from plangym import AtariEnvironment, ParallelEnvironment
    from fragile.atari import AtariEnv
    from fragile.core import DiscreteUniform, GaussianDt, Swarm

    game_name = "Qbert-ram-v0"
    env = ParallelEnvironment(
            env_class=AtariEnvironment,
            name=game_name,
            clone_seeds=True,
            autoreset=True,
            blocking=False,
        )
    dt = GaussianDt(min_dt=3, max_dt=1000, loc_dt=4, scale_dt=2)
    
    swarm = Swarm(
        model=lambda env: DiscreteUniform(env=env, critic=dt),
        env=lambda: AtariEnv(env=env),
        tree=None,
        n_walkers=64,
        max_iters=400,
        reward_scale=2,
        distance_scale=1,
        minimize=False,
    )

By default it will display the summary table, the evolution of the best
reward sampled and the best frame sampled.

.. code:: ipython3

    atviz = AtariViz(swarm, stream_interval=10)

.. code:: ipython3

    atviz.plot()

.. image::
    ../images/03_qbert.gif

.. code:: ipython3

    atviz.run()

You can display the histograms of the swarm values by passing **“all”**
to ``display_plots``.

.. code:: ipython3

    atviz = AtariViz(swarm, stream_interval=10, display_plots="all")

.. code:: ipython3

    atviz.plot()

.. image::
    ../images/03_atariviz.gif

.. code:: ipython3

    atviz.run()

