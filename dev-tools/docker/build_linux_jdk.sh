#!/bin/bash
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

# Builds the Docker image that can be used to run a subset of the ES
# integration tests.
#
# This script is not intended to be run regularly.

if [ `uname -m` != x86_64 ] ; then
    echo "Native build images must be built on the correct hardware architecture"
    echo "Required: x86_64, Current:" `uname -m`
    exit 1
fi

DOCKER_DIR=`docker info 2>/dev/null | grep '^ *Docker Root Dir' | awk -F: '{ print $2 }' | sed 's/^ *//'`
echo "Building this image may require up to 50GB of space for Docker"
echo "Current space available in $DOCKER_DIR"
df -h "$DOCKER_DIR"
sleep 5

HOST=docker.elastic.co
ACCOUNT=ml-dev
REPOSITORY=ml_cpp_jdk17
VERSION=1

set -e

cd `dirname $0`

. ./prefetch_docker_image.sh
CONTEXT=ml_cpp_jdk
prefetch_docker_base_image $CONTEXT/Dockerfile
docker build --no-cache -t $HOST/$ACCOUNT/$REPOSITORY:$VERSION $CONTEXT
# Get a username and password for this by visiting
# https://docker-auth.elastic.co and allowing it to authenticate against your
# GitHub account
docker login $HOST
docker push $HOST/$ACCOUNT/$REPOSITORY:$VERSION

