.. include:: ../color_roles.rst
Installation
========================

Fragile has been tested in Ubuntu 18.04 and Ubuntu 19.04. It supports Python 3.6, 3.7 and 3.8.
If you find any problems running it in a different OS or Python version please `open an issue <https://github.com/FragileTech/fragile/issues>`_.

Installing from pip
^^^^^^^^^^^^^^^^^^^^^

Fragile can be installed from ``pip`` running

.. code-block:: bash

    pip install fragile

This will install the package and the dependencies for using the ``core`` and
``optimize`` modules. It is also possible to install with pip the dependencies for using the
other modules

You can install the dependencies for running Atari games with:

``pip install fragile["atari"]``

The dependencies for the ``dataviz`` module can be installed running:

``pip install fragile["dataviz"]``

To install the dependencies for the ``distributed`` module you can run:

``pip install fragile["ray"]``

The dependencies for running the tests can be installed with:

``pip install fragile["test"]``

All the dependencies can be installed at once running:

.. code-block:: bash

    pip install fragile["all"]

Installing from source
^^^^^^^^^^^^^^^^^^^^^^^^

To install ``fragile`` from source you can run:

.. code-block:: bash

   git clone https://github.com/FragileTech/fragile.git
   cd fragile
   pip3 install -r requirements.txt
   pip3 install -r requirements-viz.txt
   pip3 install -e .["all"]


Running in Docker
^^^^^^^^^^^^^^^^^^
The fragile docker container will execute a Jupyter notebook accessible on port 8080 with password: `fragile`

Pulling the Docker image from Docker Hub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can pull a docker image from Docker Hub running:

.. code-block:: bash

    docker pull fragiletech/fragile:version-tag

Where version-tag corresponds to the fragile version that will be installed in the pulled image.

Building the Docker image from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After cloning the repository run:

.. code-block:: bash

   make docker build
   docker run -d -p 8080:8080 -v PATH_TO_REPO/fragile fragile


You can also run the tests inside the docker container
.. code-block:: bash

    make docker-test


You can change the default jupyter notebook password by passing ``JUPYTER_PASSWORD=your_password``
as a docker build argument.
