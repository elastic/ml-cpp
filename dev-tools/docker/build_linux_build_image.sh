#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Builds the Docker image that can be used to compile the machine learning
# C++ code for Linux.
#
# This script is not intended to be run regularly.  When changing the tools
# or 3rd party components required to build the machine learning C++ code
# increment the version, change the Dockerfile and build a new image to be
# used for subsequent builds on this branch.  Then update the version to be
# used for builds in docker/linux_builder/Dockerfile.

ACCOUNT=droberts195
REPOSITORY=ml-linux-build
VERSION=2

set -e

cd `dirname $0`

docker build -t $ACCOUNT/$REPOSITORY:$VERSION linux_image
docker login
docker push $ACCOUNT/$REPOSITORY:$VERSION

