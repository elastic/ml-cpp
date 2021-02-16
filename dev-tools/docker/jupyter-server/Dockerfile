#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#


FROM mlcpp-build:latest

LABEL maintainer="Valeriy Khakhutskyy <valeriy.khakhutskyy@elastic.co>"

WORKDIR /root

COPY ./requirements.txt /root/
COPY ./gcsfuse.repo /etc/yum.repos.d/

RUN \
    yum install -y python3-pip python3-devel python3-virtualenv openssl gcsfuse && \
    yum groupinstall -y 'Development Tools' 

RUN \
    python3 -m venv env && \
    source env/bin/activate && \
    pip install -r /root/requirements.txt && \
    jupyter notebook --generate-config && \
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 -subj "/C=US/ST=Denial/L=Springfield/O=Dis/CN=www.example.com" -keyout mykey.key -out mycert.pem 

COPY ./jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
COPY ./init.sh /root/

ENTRYPOINT ["/root/env/bin/jupyter", "notebook"]