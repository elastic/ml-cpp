#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# The non-Windows part of ML C++ CI does the following:
#
# 1. If this is not a PR build nor a debug build, obtain credentials from Vault
#    for the accessing S3
# 2. Build and unit test the Linux version of the C++
# 3. For Linux PR builds, also run some Java integration tests using the newly
#    built C++ code
# 4. If this is not a PR build nor a debug build, upload the builds to the
#    artifacts directory on S3 that subsequent Java builds will download the C++
#    components from
#
# The steps run in Docker containers that ensure OS dependencies
# are appropriate given the support matrix.

: "${HOME:?Need to set HOME to a non-empty value.}"
: "${WORKSPACE:?Need to set WORKSPACE to a non-empty value.}"

set +x

# Change directory to the directory containing this script
cd "$(dirname $0)"

# If this isn't a PR build or a debug build then obtain credentials from Vault
if [[ -z "$PR_AUTHOR" && -z "$ML_DEBUG" ]] ; then
    . ./aws_creds_from_vault.sh
fi

set -e

# Default to a snapshot build
if [ -z "$BUILD_SNAPSHOT" ] ; then
    BUILD_SNAPSHOT=true
fi

# Default to running tests
if [ -z "$RUN_TESTS" ] ; then
    RUN_TESTS=true
fi

VERSION=$(cat ../gradle.properties | grep '^elasticsearchVersion' | awk -F= '{ print $2 }' | xargs echo)
HARDWARE_ARCH=$(uname -m)

if [ "$HARDWARE_ARCH" != x86_64 ] ; then
    echo "$VERSION is not built on $HARDWARE_ARCH"
    exit 0
fi

# Jenkins sets BUILD_SNAPSHOT, but our Docker scripts require SNAPSHOT
if [ "$BUILD_SNAPSHOT" = false ] ; then
    export SNAPSHOT=no
else
    export SNAPSHOT=yes
    VERSION=${VERSION}-SNAPSHOT
fi

# Version qualifier can't be used in this branch
if [ -n "$VERSION_QUALIFIER" ] ; then
    echo "VERSION_QUALIFIER not supported on this branch: was $VERSION_QUALIFIER"
    exit 2
fi

# Tests must be run in PR builds
if [[ -n "$PR_AUTHOR" && "$RUN_TESTS" = false ]] ; then
    echo "RUN_TESTS should not be false PR builds"
    exit 3
fi

# Remove any old builds
rm -rf ../builds

# Disassociate from reference repo
git repack -a -d
readonly GIT_TOPLEVEL=$(git rev-parse --show-toplevel 2> /dev/null)
rm -f "${GIT_TOPLEVEL}/.git/objects/info/alternates"

# The Docker version is helpful to identify version-specific Docker bugs
docker --version

# Build and test Linux
if [ "$RUN_TESTS" = false ] ; then
    ./docker_build.sh linux
else
    ./docker_test.sh linux
fi

# If this is a PR build then run some Java integration tests
if [ -n "$PR_AUTHOR" ] ; then
    if [ "$(uname -s)" = Linux ] ; then
        IVY_REPO="${GIT_TOPLEVEL}/../ivy"
        mkdir -p "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION"
        cp "../build/distributions/ml-cpp-$VERSION-linux-x86_64.zip" "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION/ml-cpp-$VERSION.zip"
        ./run_es_tests.sh "${GIT_TOPLEVEL}/.." "$(cd "${IVY_REPO}" && pwd)"
    else
        echo 'Not running ES integration tests on non-Linux platform:' $(uname -a)
    fi
fi

# If this isn't a PR build and isn't a debug build then upload the artifacts
if [[ -z "$PR_AUTHOR" && -z "$ML_DEBUG" ]] ; then
    (cd .. && ./gradlew --info -b upload.gradle -Dbuild.snapshot=$BUILD_SNAPSHOT upload)
fi

