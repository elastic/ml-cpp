#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

FROM alpine:3.8

# Docker image containing the correct clang-format for check-style.sh

MAINTAINER Valeriy Khakhutskyy <valeriy.khakhutskyy@elastic.co>

RUN apk update && \
    apk add --no-cache clang bash git

WORKDIR /ml-cpp
ENV CPP_SRC_HOME=/ml-cpp
    
ENTRYPOINT ["/ml-cpp/dev-tools/clang-format.sh"]
