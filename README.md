# Fragile
[![Travis build status](https://travis-ci.org/guillemdb/fragile.svg)](https://travis-ci.org/guillemdb/fragile)
[![Code coverage](https://codecov.io/github/guillemdb/fragile/coverage.svg)](https://codecov.io/github/guillemdb/fragile)
[![Docker build status](https://img.shields.io/docker/build/guillemdb/fragile.svg)](https://hub.docker.com/r/guillemdb/fragile)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![unstable](http://badges.github.io/stability-badges/dist/unstable.svg)](http://github.com/badges/stability-badges)

**This repository is under active development.**

Fragile is a framework for developing algorithms inspired by the Fractal AI theory and testing them at scale.

## Getting started 

Check out the [Getting started with Atari games](https://guillemdb.github.io/fragile/resources/getting_started.html) 
section of the docs, or check out the examples folder.

## Installation
```bash
   git clone https://github.com/Guillemdb/fragile.git
   cd fragile
   sudo pip3 install -r requirements.txt
   sudo pip3 install -e .
```

## Running on docker
The docker container will execute a Jupyter notebook accessible on port 8080 with password: `fragile`

```bash
   docker build -t fragile .
   docker run -d -p 8080:8080 -v PATH_TO_REPO/fragile fragile 
```

## Documentation

You can access the documentation on [GitHub Pages](https://guillemdb.github.io/fragile/).

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

- Add example notebook for function optimization using the `optimize` module
- Update CI and publish docs on ReadTheDocs.org
- Document and test the `dataviz` module. Write example notebook
- Document and test the `ray` module. Write example notebook
- Document and refactor Montezuma solver
- Improve the tests (currently 154 for the `core` and `optimize` modules)
- Get rid of the `experimental` module
- Add a module to generate data for training deep learning models
- Add a benchmarking module
- Add deep learning API