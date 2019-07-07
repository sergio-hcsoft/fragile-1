# Fragile
[![Travis build status](https://travis-ci.com/guillemdb/fragile.svg)](https://travis-ci.com/guillemdb/fragile)
[![Code coverage](https://codecov.io/github/guillemdb/fragile/coverage.svg)](https://codecov.io/github/guillemdb/fragile)
[![Docker build status](https://img.shields.io/docker/build/guillemdb/fragile.svg)](https://hub.docker.com/r/guillemdb/fragile)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) 
![stability: alpha](https://svg-badge.appspot.com/badge/stability/alpha?color=f47142)

Framework for developing algorithms inspired by the Fractal AI theory.

## Installation
```bash
   git clone https://github.com/Guillemdb/fragile.git
   cd fragile
   sudo pip3 install -r requirements.txt
   sudo pip3 install -e .
```

## Documentation


* Building the documentation:
    
```bash
  cd fragile/docs
  make html
``` 

* Accessing the documentation:
    - Launch an http server:
    ```bash
      cd build/html # assuming you are inside fragile/docs
      python3 -m http.server      
    ```
    - Visit [http://0.0.0.0:8000](http://0.0.0.0:8000) to display the documentation.