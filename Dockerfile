FROM ubuntu:18.04
ENV BROWSER=/browser \
    LC_ALL=en_US.UTF-8 \
    JUPYTER_PASSWORD=fragile_in_the_cloud
COPY requirements.txt fragile/requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends \
      ca-certificates locales pkg-config apt-utils gcc g++ wget make git cmake libffi-dev \
      libjpeg-turbo-progs libglib2.0-0 python3 python3-dev python3-distutils python3-setuptools \
      libjpeg8-dev zlib1g zlib1g-dev libsm6 libxext6 libxrender1 libfontconfig1 pkg-config flex \
      bison curl libpng16-16 libpng-dev libjpeg-turbo8 libjpeg-turbo8-dev zlib1g-dev libhdf5-100 \
      libhdf5-dev libopenblas-base libopenblas-dev gfortran libfreetype6 libfreetype6-dev && \
    ln -s /usr/lib/x86_64-linux-gnu/libz.so /lib/ && \
    ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /lib/ && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    wget -O - https://bootstrap.pypa.io/get-pip.py | python3 && \
    rm -rf /var/lib/apt/lists/* && \
    echo '#!/bin/bash\n\
\n\
echo\n\
echo "  $@"\n\
echo\n\' > /browser && \
    chmod +x /browser


# install FractalAI deps
ENV NPY_NUM_BUILD_JOBS 8
RUN pip3 install --no-cache-dir cython && \
    pip3 install --no-cache-dir \
        git+https://github.com/openai/gym \
        networkx jupyter h5py Pillow-simd PyOpenGL matplotlib && \
    git clone https://github.com/ray-project/ray.git && \
    pip3 install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.8.0.dev6-cp36-cp36m-manylinux1_x86_64.whl && \
    git clone https://github.com/Guillemdb/plangym.git && \
    cd plangym && pip3 install -e . && cd .. && \
    cd fragile && \
    pip3 install -U --no-cache-dir -r requirements.txt --no-use-pep517&& \
    python3 -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot"


COPY . fragile/

RUN cd fragile && pip3 install -e . --no-use-pep517 && pip3 install jupyter psutil setproctitle && \
    pip3 uninstall -y atari-py && pip3 install git+https://github.com/Guillem-db/atari-py

RUN pip3 uninstall -y cython && \
    apt-get remove -y cmake pkg-config flex bison curl libpng-dev \
        libjpeg-turbo8-dev zlib1g-dev libhdf5-dev libopenblas-dev gfortran \
        libfreetype6-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /root/.jupyter && \
    echo 'c.NotebookApp.token = "'${JUPYTER_PASSWORD}'"' > /root/.jupyter/jupyter_notebook_config.py
CMD jupyter notebook --allow-root --port 8080 --ip 0.0.0.0