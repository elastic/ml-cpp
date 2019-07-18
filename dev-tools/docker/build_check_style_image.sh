#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Builds the Docker image that contains the correct version of clang-format for
# checking the code style.
#
# This script is not intended to be run regularly.  When changing the
# clang-format version, increment the image version, change the Dockerfile and
# build a new image to be used for subsequent builds on this branch.

HOST=push.docker.elastic.co
ACCOUNT=ml-dev
REPOSITORY=ml-check-style
VERSION=2

set -e

cd `dirname $0`

docker build --no-cache -t $HOST/$ACCOUNT/$REPOSITORY:$VERSION check_style_image
# Get a username and password for this by visiting
# https://docker.elastic.co:7000 and allowing it to authenticate against your
# GitHub account
docker login $HOST
docker push $HOST/$ACCOUNT/$REPOSITORY:$VERSION

