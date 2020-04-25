#!/bin/sh
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Script to get the appropriate version of Eigen, if not already present.
#
# If updating this script ensure the license information is correct in the
# licenses sub-directory.

cd `dirname "$BASH_SOURCE"`

# This is the file where Eigen stores its version
VERSION_FILE=eigen/Eigen/src/Core/util/Macros.h

# We want Eigen version 3.3.7 for our current branch
grep '^#define EIGEN_WORLD_VERSION 3' "$VERSION_FILE" > /dev/null 2>&1 && \
grep '^#define EIGEN_MAJOR_VERSION 3' "$VERSION_FILE" > /dev/null 2>&1 && \
grep '^#define EIGEN_MINOR_VERSION 7' "$VERSION_FILE" > /dev/null 2>&1
if [ $? -ne 0 ] ; then
    rm -rf eigen
    git -c advice.detachedHead=false clone --depth=1 --branch=3.3.7 https://gitlab.com/libeigen/eigen.git
fi

