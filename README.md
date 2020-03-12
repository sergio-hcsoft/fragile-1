# Fragile
[![Travis build status](https://travis-ci.com/FragileTech/fragile.svg)](https://travis-ci.com/FragileTech/fragile)
[![Code coverage](https://codecov.io/github/FragileTech/fragile/coverage.svg)](https://codecov.io/github/FragileTech/fragile)
[![PyPI package](https://badgen.net/pypi/v/fragile)](https://pypi.org/project/fragile/)
[![Latest docker image](https://badgen.net/docker/pulls/fragiletech/fragile)](https://hub.docker.com/r/fragiletech/fragile/tags)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![license: AGPL v3](https://img.shields.io/badge/license-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![unstable](https://badgen.net/badge/stability/unstable/E5AE13)](http://github.com/badges/stability-badges)

**This repository is under active development.**

Fragile is a framework for developing algorithms inspired by the Fractal AI theory and testing them at scale.

## About FractalAI

FractalAI is based on the framework of non-equilibrium thermodynamics, and It allows to derive new 
mathematical tools for efficiently exploring state spaces.
 
 The principles of our work are accessible online:

- [Arxiv](https://arxiv.org/abs/1803.05049) manuscript describing the fundamental principles of our work.
- [Blog](http://entropicai.blogspot.com) that describes our early research process.
- [Youtube channel](https://www.youtube.com/user/finaysergio/videos) with videos showing how different prototypes work.
- [GitHub repository](https://github.com/FragileTech/FractalAI) containing a prototype that solves most Atari games.

## Getting started 

Check out the [Getting started with Atari games](https://fragiletech.github.io/fragile/resources/examples/examples_index.html#getting-started) 
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

It can be install with `pip install fragile`.

### Building from source

Please take a look at the Dockerfile to find out about all the dependencies, and the detailed 
installation process.

```bash
   git clone https://github.com/FragileTech/fragile.git
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

- Document and test the `ray` module. Write example notebook
- Document and refactor Montezuma solver
- Improve tests coverage (currently 153 for the `core` and `optimize` modules)
- Add new algorithms to sample different state spaces.
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
