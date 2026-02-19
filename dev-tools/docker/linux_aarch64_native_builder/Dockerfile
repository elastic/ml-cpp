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
FROM docker.elastic.co/ml-dev/ml-linux-aarch64-native-build:17

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Copy the current Git repository into the container
COPY . /ml-cpp/

ENV CMAKE_FLAGS -DCMAKE_TOOLCHAIN_FILE=/ml-cpp/cmake/linux-aarch64.cmake

# Pass through any version qualifier (default none)
ARG VERSION_QUALIFIER=

# Pass through whether this is a snapshot build (default yes if not specified)
ARG SNAPSHOT=yes

# Pass through ML debug option (default blank)
ARG ML_DEBUG=

# Zip compression level (1=fast for CI, 9=max for release)
ARG ZIP_COMPRESSION_LEVEL=9

# Pass through the format of the unit test output files
ARG BOOST_TEST_OUTPUT_FORMAT_FLAGS=

# sccache with GCS backend (optional — leave SCCACHE_GCS_BUCKET empty to disable)
ARG SCCACHE_GCS_BUCKET=
ENV SCCACHE_GCS_BUCKET=$SCCACHE_GCS_BUCKET

# Run the build
RUN --mount=type=secret,id=gcs_key,required=false \
  if [ -f /run/secrets/gcs_key ]; then export SCCACHE_GCS_KEY_PATH=/run/secrets/gcs_key; fi && \
  /ml-cpp/dev-tools/docker/docker_entrypoint.sh

