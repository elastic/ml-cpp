#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Checks the formatting of the machine learning C++ code in a Docker container.
# (This avoids the need to have the correct version of clang-format installed
# locally.)
#
# The Docker container into which the code is copied is deleted after the check.

# The build needs to be done with the Docker context set to the root of the
# repository so that we can copy it into the container.
MY_DIR=`dirname "$BASH_SOURCE"`
TOOLS_DIR=`cd "$MY_DIR" && pwd`

# The Docker context here is the root directory of the outer repository.
cd "$TOOLS_DIR/.."

# This Dockerfile is for the temporary image that is used to do the style check.
# It is based on a pre-built image stored on Docker Hub, but will have the local
# repository contents copied into it before the check-style.sh script is run.
# This temporary image is discarded after the check is complete.
DOCKERFILE="$TOOLS_DIR/docker/style_checker/Dockerfile"
TEMP_TAG=`git rev-parse --short=14 HEAD`-style-$$

docker build --no-cache --force-rm -t $TEMP_TAG -f "$DOCKERFILE" .
docker run --rm --workdir=/ml-cpp $TEMP_TAG dev-tools/check-style.sh --all
RC=$?
docker rmi --force $TEMP_TAG
exit $RC

