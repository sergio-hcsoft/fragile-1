# Fragile
[![Travis build status](https://travis-ci.org/guillemdb/fragile.svg)](https://travis-ci.org/guillemdb/fragile)
[![Code coverage](https://codecov.io/github/guillemdb/fragile/coverage.svg)](https://codecov.io/github/guillemdb/fragile)
[![Docker build status](https://img.shields.io/docker/build/guillemdb/fragile.svg)](https://hub.docker.com/r/guillemdb/fragile)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![unstable](http://badges.github.io/stability-badges/dist/unstable.svg)](http://github.com/badges/stability-badges)

**This repository is under active development. I am actively working to make it easy to install and test.**

Fragile is a framework for developing algorithms inspired by the Fractal AI theory and testing them at scale.

## About FractalAI

FractalAI is a theory of Artificial Intelligence derived from first principles, and based on the 
framework of non-equilibrium thermodynamics. It allows to derive new mathematical tools for efficiently
 exploring state spaces.
 
 The fundamental principles of our work are accessible online:

- [Arxiv](https://arxiv.org/abs/1803.05049) manuscript describing the fundamental principles of our work.
- [Blog](http://entropicai.blogspot.com) that describes our early research process.
- [Youtube channel](https://www.youtube.com/user/finaysergio/videos) with videos showing how different prototypes work.
- [GitHub repository](https://github.com/FragileTech/FractalAI) containing a prototype that solves most Atari games.

## Getting started 

Check out the [Getting started with Atari games](https://fragiletech.github.io/fragile/resources/getting_started.html) 
section of the docs, or check out the examples folder.

## Running in docker
The docker container will execute a Jupyter notebook accessible on port 8080 with password: `fragile`

```bash
   make docker build
   docker run -d -p 8080:8080 -v PATH_TO_REPO/fragile fragile 
```

You can also run the tests inside the docker container
```bash
    make docker-test
```

## Installation
This framework has been tested in Ubuntu 18.04 and supports Python 3.6, 3.7 and 3.8. 
If you find any problems running it in a different OS or Python version please open an issue.

Please take a look at the Dockerfile to find out about all the dependencies, and the detailed installation process.

```bash
   git clone https://github.com/Guillemdb/fragile.git
   cd fragile
   pip3 install -r requirements.txt
   pip3 install -r requirements-viz.txt
   pip3 install -e .
```

## Documentation

You can access the documentation on [GitHub Pages](https://fragiletech.github.io/fragile/).

* Building the documentation:
    
```bash
  cd fragile/docs
  make html
``` 

* Accessing the documentation locally:
    - Launch an http server:
    ```bash
      cd build/html # assuming you are inside fragile/docs
      python3 -m http.server      
    ```
    - Visit [http://0.0.0.0:8000](http://0.0.0.0:8000) to display the documentation.
    
## Roadmap

- Publish docs on ReadTheDocs.org
- Document and test the `dataviz` module. Write example notebook
- Document and test the `ray` module. Write example notebook
- Document and refactor Montezuma solver
- Improve tests coverage (currently 153 for the `core` and `optimize` modules)
- Add a module to generate data for training deep learning models
- Add a benchmarking module
- Add deep learning API

## Contributing

Contribution are welcome. Please take a look at [contributining](docsrc/markdown/CONTRIBUTING.md) 
and respect the [code of conduct](docsrc/markdown/CODE_OF_CONDUCT.md).
    
## Cite us
If you use this framework in your research please cite us as:

    @misc{1803.05049,
        Author = {Sergio Hernández Cerezo and Guillem Duran Ballester},
        Title = {Fractal AI: A fragile theory of intelligence},
        Year = {2018},
        Eprint = {arXiv:1803.05049},
      }
      
## License

This project is currently licensed under AGPLv3.0. 

However, if you are considering using it for applications that require a more permissive license, 
please let me know in this [Issue](https://github.com/Guillemdb/fragile/issues/5)