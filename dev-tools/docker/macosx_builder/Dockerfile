#
# ELASTICSEARCH CONFIDENTIAL
#
# Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
#
# Notice: this software, and all information contained
# therein, is the exclusive property of Elasticsearch BV
# and its licensors, if any, and is protected under applicable
# domestic and foreign law, and international treaties.
#
# Reproduction, republication or distribution without the
# express written consent of Elasticsearch BV is
# strictly prohibited.
#

# Increment the version here when a new tools/3rd party components image is built
FROM droberts195/ml-macosx-build:3

MAINTAINER David Roberts <dave.roberts@elastic.co>

# Copy the current Git repository into the container
COPY . /ml-cpp/

# Tell the build we want to cross compile
ENV CPP_CROSS_COMPILE macosx

# Pass through whether this is a snapshot build (default yes if not specified)
ARG SNAPSHOT=yes

# Run the build
RUN \
  /ml-cpp/dev-tools/docker/docker_entrypoint.sh

