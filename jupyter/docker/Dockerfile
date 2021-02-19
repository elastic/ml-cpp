#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
FROM docker.elastic.co/ml-dev/ml-linux-build:15 as builder 

RUN yum install -y epel-release && \
    yum install -y ccache
ENV PATH="/usr/lib64/ccache:${PATH}"

COPY . /ml-cpp
WORKDIR /ml-cpp
ENV CPP_SRC_HOME=/ml-cpp
RUN --mount=type=cache,target=/root/.ccache . ./set_env.sh  && make -j`nproc`
##############################################
FROM centos:7

USER root
WORKDIR /root

COPY jupyter/docker/init.sh /root/
COPY jupyter/requirements.txt /root/
COPY jupyter/docker/gcsfuse.repo /etc/yum.repos.d/

RUN \
    yum install -y epel-release && \
    yum install -y python3-pip python3-devel python3-virtualenv openssl gcsfuse htop tmux && \
    yum groupinstall -y 'Development Tools' && \
    python3 -m venv env && \
    source env/bin/activate && \
    pip install -r /root/requirements.txt && \
    jupyter notebook --generate-config && \
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 -subj "/C=US/ST=Denial/L=Springfield/O=Dis/CN=www.example.com" -keyout mykey.key -out mycert.pem && \
    chmod 700 /root/init.sh

COPY jupyter/docker/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

COPY --from=builder /ml-cpp/build/distribution/platform/linux-x86_64 /ml-cpp
COPY ./jupyter/notebooks /root/notebooks

ENTRYPOINT ["/root/env/bin/jupyter", "notebook"]