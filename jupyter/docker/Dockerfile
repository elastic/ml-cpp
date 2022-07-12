# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# syntax=docker/dockerfile:1.3
FROM docker.elastic.co/ml-dev/ml-linux-build:24 as builder

RUN yum install -y epel-release && \
    yum install -y ccache && \
    ln -s /bin/ccache /usr/lib64/ccache/g++
ENV PATH="/usr/lib64/ccache:${PATH}"
ENV CCACHE_SLOPPINESS="file_stat_matches,pch_defines,time_macros,include_file_mtime,include_file_ctime,system_headers"

COPY . /ml-cpp
WORKDIR /ml-cpp
ENV CPP_SRC_HOME=/ml-cpp
ENV LD_LIBRARY_PATH=/usr/local/gcc103/lib/:/usr/local/gcc103/lib64/

RUN --mount=type=cache,target=/root/.ccache \
    . ./set_env.sh &&  \
    cmake -B cmake-build-relwithdebinfo && \
    cmake --build cmake-build-relwithdebinfo --config Release -t install -j`nproc` && \
    find build/distribution/platform/linux-x86_64/lib ! -name libMl*.so -type f -exec rm -f {} + && \
    dev-tools/strip_binaries.sh 


##############################################
FROM docker.elastic.co/ml-dev/ml-linux-jupyter-build:1
 
COPY --from=builder /usr/local/gcc103 /usr/local/gcc103
ENV LD_LIBRARY_PATH=/usr/local/gcc103/lib/:/usr/local/gcc103/lib64/

COPY jupyter/requirements.txt jupyter/Makefile /root/
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /root && \
    python3 -m venv env && \
    source env/bin/activate && \
    python -m pip install -U --force-reinstall pip && \
    pip install -r requirements.txt && \
    jupyter notebook --generate-config && \
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 -subj "/C=US/ST=Denial/L=Springfield/O=Dis/CN=www.example.com" -keyout mykey.key -out mycert.pem

COPY jupyter/config.ini  /root/
COPY jupyter/docker/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
COPY jupyter/notebooks /root/notebooks
COPY jupyter/src /root/src
RUN  source env/bin/activate &&  pip install -e /root/src
COPY jupyter/scripts /root/scripts

COPY --from=builder /ml-cpp/build/distribution/platform/linux-x86_64 /ml-cpp

ENTRYPOINT ["/root/env/bin/jupyter", "notebook"]
