import os
from setuptools import find_packages, setup

from fragile.version import __version__

import os
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    install_requires = []
    packages = [],
else:
    install_requires = ["numpy", "scipy", "networkx", "atari-py==0.1.1"]
    packages = find_packages()

setup(
    name="fragile",
    description="Framework for developing FractalAI based algorithms.",
    packages=packages,
    version=__version__,
    license="AGPLv3.0",
    author="Guillem Duran Ballester",
    author_email="guillem.db@gmail.com",
    url="https://github.com/Guillemdb/fragile",
    download_url="https://github.com/Guillemdb/fragile",
    keywords=["reinforcement learning", "artificial intelligence", "monte carlo", "planning"],
    test_requires=["pytest", "hypothesis"],
    install_requires=install_requires,
    package_data={"": ["README.md"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: AGPLv3.0",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries",
    ],
)
