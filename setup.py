from importlib.machinery import SourceFileLoader
from pathlib import Path

from setuptools import find_packages, setup


version = SourceFileLoader(
    "fragile.version",
    str(Path(__file__).parent / "fragile" / "version.py"),
).load_module()


setup(
    name="fragile",
    description="Framework for developing FractalAI based algorithms.",
    packages=find_packages(),
    version=version.__version__,
    license="AGPLv3.0",
    author="Guillem Duran Ballester",
    author_email="guillem.db@gmail.com",
    url="https://github.com/Guillemdb/fragile",
    download_url="https://github.com/Guillemdb/fragile",
    keywords=["reinforcement learning", "artificial intelligence", "monte carlo", "planning"],
    tests_require=["pytest", "hypothesis"],
    install_requires=["numpy", "scipy", "networkx", "atari-py==0.1.1"],
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
