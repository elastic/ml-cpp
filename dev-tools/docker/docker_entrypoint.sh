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

# This script gets run within the Docker container when a build is done in a
# Docker container.
#
# It is not intended to be run outside of a Docker container (although it
# should work if it is).

set -e

if [ -z "$CMAKE_QUIET" ]; then
  CMAKE_VERBOSE="-v"
fi

# Change directory to the root of the Git repository
MY_DIR=`dirname "$BASH_SOURCE"`
cd "$MY_DIR/../.."

# Set a consistent environment
. ./set_env.sh

# Shared helper asserting the controller-protocol.version marker is packaged.
. ./dev-tools/verify_controller_protocol_version.sh

# Note: no need to clean due to the .dockerignore file

# Configure the build
cmake -B cmake-build-docker ${CMAKE_FLAGS}

# Build the code
cmake --build cmake-build-docker ${CMAKE_VERBOSE} -j`nproc` -t install

# Strip the binaries
dev-tools/strip_binaries.sh

# Get the version number
PRODUCT_VERSION=`cat "$CPP_SRC_HOME/gradle.properties" | grep '^elasticsearchVersion' | awk -F= '{ print $2 }' | xargs echo`
if [ -n "$VERSION_QUALIFIER" ] ; then
    PRODUCT_VERSION="$PRODUCT_VERSION-$VERSION_QUALIFIER"
fi
if [ "$SNAPSHOT" = yes ] ; then
    PRODUCT_VERSION="$PRODUCT_VERSION-SNAPSHOT"
fi

ARTIFACT_NAME=`cat "$CPP_SRC_HOME/gradle.properties" | grep '^artifactName' | awk -F= '{ print $2 }' | xargs echo`

# Create the output artifacts
cd build/distribution
mkdir -p ../distributions
# Publish the controller protocol/capability marker at the bundle root so
# Elasticsearch's verifyControllerProtocolVersion gate can assert against it
# in the -nodeps bundle. The Gradle buildZip task already does this for the
# macOS/Gradle packaging path; this is the equivalent for the Linux Docker
# packaging path, which is what CI's Elasticsearch Java integration tests
# (and create_dra.sh) actually resolve. 'set -e' above means a missing
# marker source fails the build here rather than downstream.
cp "$CPP_SRC_HOME/3rd_party/controller-protocol.version" .
# Exclude import libraries, test support libraries, debug files and core dumps
zip -9 ../distributions/$ARTIFACT_NAME-$PRODUCT_VERSION-$BUNDLE_PLATFORM.zip `find * | egrep -v '\.lib$|unit_test_framework|libMlTest|\.dSYM|-debug$|\.pdb$|/core'`
# Include only debug files
zip -9 ../distributions/$ARTIFACT_NAME-$PRODUCT_VERSION-debug-$BUNDLE_PLATFORM.zip `find * | egrep '\.dSYM|-debug$|\.pdb$'`
verify_controller_protocol_version ../distributions/$ARTIFACT_NAME-$PRODUCT_VERSION-$BUNDLE_PLATFORM.zip || exit 1
cd ../..

if [ "x$1" = "x--test" ] ; then
    # Convert any failure of this make command into the word passed or failed in
    # a status file - this allows the Docker image build to succeed if the only
    # failure is the unit tests, and then the detailed test results can be
    # copied from the image
    echo passed > build/test_status.txt
    cmake --build cmake-build-docker ${CMAKE_VERBOSE} -j`nproc` -t test || echo failed > build/test_status.txt
fi

