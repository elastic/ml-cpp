#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Increment the version here when a new tools/3rd party components image is built
FROM docker.elastic.co/ml-dev/ml-linux-aarch64-native-build:1

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Copy the current Git repository into the container
COPY . /ml-cpp/

# Pass through any version qualifier (default none)
ARG VERSION_QUALIFIER=

# Pass through whether this is a snapshot build (default yes if not specified)
ARG SNAPSHOT=yes

# Run the build and unit tests
RUN \
  /ml-cpp/dev-tools/docker/docker_entrypoint.sh --test

