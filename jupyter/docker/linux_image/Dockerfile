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
    yum install -y python3-pip python3-devel python3-virtualenv openssl htop tmux && \
    yum groupinstall -y 'Development Tools' && \
    curl -L -o ts-1.0.1.tar.gz http://viric.name/soft/ts/ts-1.0.1.tar.gz && \
    tar -xzf ts-1.0.1.tar.gz && \
    cd ts-1.0.1 && \
    make && make install PREFIX=/usr && \
    ln -s /usr/bin/ts /usr/bin/tsp && \
    tsp -S `nproc` && \
    mkdir -p /root/data/configs /root/data/datasets /root/data/jobs