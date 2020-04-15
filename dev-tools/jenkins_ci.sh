#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# The non-Windows part of ML C++ CI does the following:
#
# 1. If this is not a PR build, obtain credentials from Vault for the accessing
#    S3
# 2. If this is a PR build and running on x86_64, check the code style
# 3. Build and unit test the Linux version of the C++ on the native architecture
# 4. If running on x86_64, cross compile the macOS build of the C++
# 5. If this is not a PR build and running on x86_64, cross compile the aarch
#    build of the C++
# 6. If this is not a PR build, upload the builds to the artifacts directory on
#    S3 that subsequent Java builds will download the C++ components from
#
# The steps run in Docker containers that ensure OS dependencies
# are appropriate given the support matrix.
#
# The macOS build cannot be unit tested as it is cross-compiled.

: "${HOME:?Need to set HOME to a non-empty value.}"
: "${WORKSPACE:?Need to set WORKSPACE to a non-empty value.}"

# If this isn't a PR build then obtain credentials from Vault
if [ -z "$PR_AUTHOR" ] ; then
    set +x
    export VAULT_TOKEN=$(vault write -field=token auth/approle/login role_id="$VAULT_ROLE_ID" secret_id="$VAULT_SECRET_ID")

    AWS_CREDS=$(vault read -format=json -field=data aws-dev/creds/prelertartifacts)
    export ML_AWS_ACCESS_KEY=$(echo $AWS_CREDS | jq -r '.access_key')
    export ML_AWS_SECRET_KEY=$(echo $AWS_CREDS | jq -r '.secret_key')

    unset VAULT_TOKEN VAULT_ROLE_ID VAULT_SECRET_ID
    set -x
fi

set -e

# Change directory to the directory containing this script
cd "$(dirname $0)"

# Default to a snapshot build
if [ -z "$BUILD_SNAPSHOT" ] ; then
    BUILD_SNAPSHOT=true
fi

VERSION=$(cat ../gradle.properties | grep '^elasticsearchVersion' | awk -F= '{ print $2 }' | xargs echo)
HARDWARE_ARCH=$(uname -m)

# Jenkins sets BUILD_SNAPSHOT, but our Docker scripts require SNAPSHOT
if [ "$BUILD_SNAPSHOT" = false ] ; then
    export SNAPSHOT=no
else
    export SNAPSHOT=yes
    VERSION=${VERSION}-SNAPSHOT
fi

# Remove any old builds
rm -rf ../builds

# Disassociate from reference repo
git repack -a -d
readonly GIT_TOPLEVEL=$(git rev-parse --show-toplevel 2> /dev/null)
rm -f "${GIT_TOPLEVEL}/.git/objects/info/alternates"

# Build and test the native Linux architecture
if [ "$HARDWARE_ARCH" = x86_64 ] ; then

    # If this is a PR build then fail fast on style checks
    if [ -n "$PR_AUTHOR" ] ; then
        ./docker_check_style.sh
    fi

    ./docker_test.sh linux
elif [ "$HARDWARE_ARCH" = aarch64 ] ; then
    ./docker_test.sh linux_aarch64_native
fi

# If this is a PR build then run some Java integration tests
if [ -n "$PR_AUTHOR" ] ; then
    if [ "$(uname -s)" = Linux ] ; then
        IVY_REPO="${GIT_TOPLEVEL}/../ivy"
        mkdir -p "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION"
        cp "../build/distributions/ml-cpp-$VERSION-linux-$HARDWARE_ARCH.zip" "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION/ml-cpp-$VERSION.zip"
        ./run_es_tests.sh "${GIT_TOPLEVEL}/.." "$(cd "${IVY_REPO}" && pwd)"
    else
        echo 'Not running ES integration tests on non-Linux platform:' $(uname -a)
    fi
fi

# Cross compile macOS
if [ "$HARDWARE_ARCH" = x86_64 ] ; then
    ./docker_build.sh macosx
fi

# If this isn't a PR build then cross compile aarch64 and upload the artifacts
if [ -z "$PR_AUTHOR" ] ; then
    if [ "$HARDWARE_ARCH" = x86_64 ] ; then
        ./docker_build.sh linux_aarch64_cross
    fi
    cd ..
    ./gradlew --info -b upload.gradle -Dbuild.snapshot=$BUILD_SNAPSHOT upload
fi

