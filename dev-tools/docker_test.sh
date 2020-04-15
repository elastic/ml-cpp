#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# Builds the machine learning C++ code for Linux in a Docker container,
# then runs the unit tests.
#
# The output .zip files are then copied out of the container to the
# location in the current repository that they'd be in had they been
# built outside of Docker.
#
# Finally, the Docker container used for the build/test is deleted.

usage() {
    echo "Usage: $0 linux|linux_aarch64_native ..."
    exit 1
}

PLATFORMS=

while [ -n "$1" ]
do

    case "$1" in
        linux|linux_aarch64_native)
            PLATFORMS="$1 $PLATFORMS"
            ;;
        *)
            usage
            ;;
    esac

    shift

done

if [ -z "$PLATFORMS" ] ; then
    usage
fi

# Default to no version qualifier

# Default to a snapshot build
if [ -z "$SNAPSHOT" ] ; then
    SNAPSHOT=yes
fi

set -e

# The build needs to be done with the Docker context set to the root of the
# repository so that we can copy it into the container.
MY_DIR=`dirname "$BASH_SOURCE"`
TOOLS_DIR=`cd "$MY_DIR" && pwd`

# The Docker context here is the root directory of the outer repository.
cd "$TOOLS_DIR/.."

for PLATFORM in `echo $PLATFORMS | tr ' ' '\n' | sort -u`
do

    # This Dockerfile is for the temporary image that is used to do the build
    # and unit tests.  It is based on a pre-built build image stored on Docker
    # Hub, but will have the local repository contents copied into it before
    # the entrypoint script is run.  This temporary image is discarded after
    # the build and unit tests are complete.
    DOCKERFILE="$TOOLS_DIR/docker/${PLATFORM}_tester/Dockerfile"
    TEMP_TAG=`git rev-parse --short=14 HEAD`-$PLATFORM-$$

    docker build --no-cache --force-rm -t $TEMP_TAG --build-arg VERSION_QUALIFIER="$VERSION_QUALIFIER" --build-arg SNAPSHOT=$SNAPSHOT -f "$DOCKERFILE" .
    # Using tar to copy the build and test artifacts out of the container seems
    # more reliable than docker cp, and also means the files end up with the
    # correct uid/gid
    docker run --rm --workdir=/ml-cpp $TEMP_TAG bash -c 'find . -name boost_test_results.xml | xargs tar cf - build/distributions build/test_status.txt' | tar xvf -
    docker rmi --force $TEMP_TAG
    # The image build is set to return zero (i.e. succeed as far as Docker is
    # concerned) when the only problem is that the unit tests fail, as this
    # gives us the chance to copy the unit test result files out of it.  But
    # then we need to check if the test status file says the tests passed to
    # decide whether this script succeeds or fails.
    grep passed build/test_status.txt

done

