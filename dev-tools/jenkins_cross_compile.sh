#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#

# The part of ML C++ CI that cross compiles platforms that are not going to be
# natively compiled.
#
# This script must run on linux-x86_64, as that is always the host OS for our
# cross compilation.
#
# 1. If this is not a PR build nor a debug build, obtain credentials from Vault
#    for the accessing S3
# 2. If this is a PR build, check the code style
# 3. Cross compile the darwin-x86_64 build of the C++
# 4. If this is not a PR build, cross compile the linux-aarch64 build of the C++
# 5. If this is not a PR build nor a debug build, upload the builds to the
#    artifacts directory on S3 that subsequent Java builds will download the C++
#    components from
#
# All steps run in Docker containers that ensure OS dependencies are appropriate
# given the support matrix.
#
# Cross-compiled platforms cannot be unit tested.

: "${HOME:?Need to set HOME to a non-empty value.}"
: "${WORKSPACE:?Need to set WORKSPACE to a non-empty value.}"

set +x

# If this isn't a PR build or a debug build then obtain credentials from Vault
if [[ -z "$PR_AUTHOR" && -z "$ML_DEBUG" ]] ; then
    . ./aws_creds_from_vault.sh
fi

set -e

if [[ `uname` != Linux || `uname -m` != x86_64 ]] ; then
    echo "This script must be run on linux-x86_64"
    exit 1
fi

# Change directory to the directory containing this script
cd "$(dirname $0)"

# Default to a snapshot build
if [ -z "$BUILD_SNAPSHOT" ] ; then
    BUILD_SNAPSHOT=true
fi

# Jenkins sets BUILD_SNAPSHOT, but our Docker scripts require SNAPSHOT
if [ "$BUILD_SNAPSHOT" = false ] ; then
    export SNAPSHOT=no
else
    export SNAPSHOT=yes
fi

# Version qualifier can't be used in this branch
if [ -n "$VERSION_QUALIFIER" ] ; then
    echo "VERSION_QUALIFIER not supported in this branch: was $VERSION_QUALIFIER"
    exit 2
fi

# Remove any old builds
rm -rf ../builds

# Disassociate from reference repo
git repack -a -d
readonly GIT_TOPLEVEL=$(git rev-parse --show-toplevel 2> /dev/null)
rm -f "${GIT_TOPLEVEL}/.git/objects/info/alternates"

# The Docker version is helpful to identify version-specific Docker bugs
docker --version

# If this is a PR build then fail fast on style checks
if [ -n "$PR_AUTHOR" ] ; then
    ./docker_check_style.sh
fi

# Cross compile macOS
./docker_build.sh macosx

# If this isn't a PR build cross compile aarch64 too
if [ -z "$PR_AUTHOR" ] ; then
    ./docker_build.sh linux_aarch64_cross
fi

# If this isn't a PR build and isn't a debug build then upload the artifacts
if [[ -z "$PR_AUTHOR" && -z "$ML_DEBUG" ]] ; then
    (cd .. && ./gradlew --info -b upload.gradle -Dbuild.snapshot=$BUILD_SNAPSHOT upload)
fi

