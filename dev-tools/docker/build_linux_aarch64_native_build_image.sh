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

if [ `uname -m` != aarch64 ] ; then
    echo "Native build images must be built on the correct hardware architecture"
    echo "Required: aarch64, Current:" `uname -m`
    exit 1
fi

HOST=push.docker.elastic.co
ACCOUNT=ml-dev
REPOSITORY=ml-linux-aarch64-native-build
VERSION=1

set -e

cd `dirname $0`

docker build --no-cache -t $HOST/$ACCOUNT/$REPOSITORY:$VERSION linux_aarch64_native_image
# Get a username and password for this by visiting
# https://docker.elastic.co:7000 and allowing it to authenticate against your
# GitHub account
docker login $HOST
docker push $HOST/$ACCOUNT/$REPOSITORY:$VERSION

