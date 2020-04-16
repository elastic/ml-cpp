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

HOST=push.docker.elastic.co
ACCOUNT=ml-dev
REPOSITORY=ml-linux-aarch64-cross-build
VERSION=1

set -e

cd `dirname $0`

docker build --no-cache -t $HOST/$ACCOUNT/$REPOSITORY:$VERSION linux_aarch64_cross_image
# Get a username and password for this by visiting
# https://docker.elastic.co:7000 and allowing it to authenticate against your
# GitHub account
docker login $HOST
docker push $HOST/$ACCOUNT/$REPOSITORY:$VERSION

