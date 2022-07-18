# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# image name docker.elastic.co/ml-dev/ml-linux-jupyter-build:1
# syntax=docker/dockerfile:1.3
FROM centos:7

USER root
WORKDIR /root

RUN --mount=type=cache,target=/var/cache/yum \
    yum install -y epel-release && \
    yum install -y wget make gcc openssl-devel bzip2-devel openssl htop tmux libffi-devel sqlite-devel && \
    yum groupinstall -y 'Development Tools' 
RUN --mount=type=cache,target=/root/.ccache \
    curl -L -o Python-3.8.7.tgz https://www.python.org/ftp/python/3.8.7/Python-3.8.7.tgz && \
    tar -xzf Python-3.8.7.tgz && \ 
    cd Python-3.8.7 && \
    ./configure --enable-optimizations && \
    make -j altinstall && \
    ln -sfn /usr/local/bin/python3.8 /usr/bin/python3 && \
    ln -sfn /usr/local/bin/pip3.8 /usr/bin/pip3 && \
    cd .. && rm -rf Python-3.8.7 
RUN curl -L -o ts-1.3.1.tar.gz https://github.com/justanhduc/task-spooler/archive/refs/tags/v1.3.1-cpu.tar.gz && \
    tar -xzf ts-1.3.1.tar.gz && \
    cd task-spooler-1.3.1-cpu && \
    make && make install PREFIX=/usr && \
    ln -s /usr/bin/ts /usr/bin/tsp && \
    tsp -S `nproc` && \
    mkdir -p /root/data/configs /root/data/datasets /root/data/jobs