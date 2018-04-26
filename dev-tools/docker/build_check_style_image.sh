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

ACCOUNT=droberts195
REPOSITORY=ml-check-style
VERSION=1

set -e

cd `dirname $0`

docker build --no-cache -t $ACCOUNT/$REPOSITORY:$VERSION check_style_image
docker login
docker push $ACCOUNT/$REPOSITORY:$VERSION

