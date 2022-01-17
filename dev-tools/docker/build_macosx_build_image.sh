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

# Builds the Docker image that can be used to compile the machine learning
# C++ code for Linux
#
# This script is not intended to be run regularly.  When changing the tools
# or 3rd party components required to build the machine learning C++ code
# increment the version, change the Dockerfile and build a new image to be
# used for subsequent builds on this branch.  Then update the version to be
# used for builds in docker/macosx_builder/Dockerfile.

HOST=docker.elastic.co
ACCOUNT=ml-dev
REPOSITORY=ml-macosx-build
VERSION=7

set -e

cd `dirname $0`

. ./prefetch_docker_image.sh
CONTEXT=macosx_image
prefetch_docker_base_image $CONTEXT/Dockerfile
docker build --no-cache -t $HOST/$ACCOUNT/$REPOSITORY:$VERSION $CONTEXT
# Get a username and password for this by visiting
# https://docker-auth.elastic.co and allowing it to authenticate against your
# GitHub account
docker login $HOST
docker push $HOST/$ACCOUNT/$REPOSITORY:$VERSION

