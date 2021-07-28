#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#

FROM alpine:3.8

# Docker image containing the correct clang-format for check-style.sh

MAINTAINER Valeriy Khakhutskyy <valeriy.khakhutskyy@elastic.co>

RUN apk update && \
    apk add --no-cache clang bash git

WORKDIR /ml-cpp
ENV CPP_SRC_HOME=/ml-cpp

