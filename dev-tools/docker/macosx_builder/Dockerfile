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

# Increment the version here when a new tools/3rd party components image is built
FROM docker.elastic.co/ml-dev/ml-macosx-build:15

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Copy the current Git repository into the container
COPY . /ml-cpp/

# Tell the build we want to cross compile
ENV CPP_CROSS_COMPILE macosx

ENV CMAKE_FLAGS -DCMAKE_TOOLCHAIN_FILE=/ml-cpp/cmake/darwin-x86_64.cmake

# Pass through any version qualifier (default none)
ARG VERSION_QUALIFIER=

# Pass through whether this is a snapshot build (default yes if not specified)
ARG SNAPSHOT=yes

# Pass through ML debug option (default blank)
ARG ML_DEBUG=

# Run the build
RUN \
  /ml-cpp/dev-tools/docker/docker_entrypoint.sh

