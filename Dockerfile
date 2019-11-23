FROM ubuntu:18.04

ENV BROWSER=/browser \
    LC_ALL=en_US.UTF-8 \
    JUPYTER_PASSWORD=fragile_in_the_cloud

COPY requirements.txt fragile/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-suggests --no-install-recommends \
      ca-certificates locales gcc g++ wget make git cmake libffi-dev libjpeg-turbo-progs libglib2.0-0 \
      python3 python3-dev python3-distutils python3-setuptools libjpeg8-dev zlib1g zlib1g-dev \
      libsm6 libxext6 libxrender1 libfontconfig1 && \
    ln -s /usr/lib/x86_64-linux-gnu/libz.so /lib/ && \
    ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /lib/ && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    wget -O - https://bootstrap.pypa.io/get-pip.py | python3 && \
    git clone https://github.com/Guillemdb/plangym.git && \
    cd plangym && pip3 install -e . && cd .. && \
    cd fragile && \
    pip3 install -U --no-cache-dir -r requirements.txt --no-use-pep517&& \
    apt-get remove -y python3-dev libxml2-dev gcc g++ wget && \
    apt-get remove -y .*-doc .*-man >/dev/null && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo '#!/bin/bash\n\
\n\
echo\n\
echo "  $@"\n\
echo\n\' > /browser && \
    chmod +x /browser

COPY . fragile/

RUN cd fragile && pip3 install -e . --no-use-pep517 && pip3 install jupyter

RUN mkdir /root/.jupyter && \
    echo 'c.NotebookApp.token = "'${JUPYTER_PASSWORD}'"' > /root/.jupyter/jupyter_notebook_config.py
CMD jupyter notebook --allow-root --port 8080 --ip 0.0.0.0