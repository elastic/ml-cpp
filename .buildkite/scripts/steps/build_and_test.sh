#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

set -euo pipefail

# Default to a snapshot build
if [ -z "$BUILD_SNAPSHOT" ] ; then
    BUILD_SNAPSHOT=true
fi

# Default to running tests.
# Not every build step will be able to run the tests
# e.g. cross compilations.
if [ -z "$RUN_TESTS" ] ; then
    RUN_TESTS=true
fi

if [ "$BUILD_SNAPSHOT" = false ] ; then
    export SNAPSHOT=no
else
    export SNAPSHOT=yes
fi

# Version qualifier shouldn't be used in PR builds
if [[ x"$BUILDKITE_PULL_REQUEST" != xfalse && -n "$VERSION_QUALIFIER" ]] ; then
    echo "VERSION_QUALIFIER should not be set in PR builds: was $VERSION_QUALIFIER"
    exit 2
fi

# If this is a PR build then it's redundant to cross compile aarch64 (as
# we build and test aarch64 natively for PR builds) but there's a benefit
# to building one platform with debug enabled to detect code that only
# compiles with optimisation
if [[ x"$BUILDKITE_PULL_REQUEST" != xfalse && "$CPP_CROSS_COMPILE" = "aarch64" ]] ; then
    export ML_DEBUG=1
fi

# For now, re-use our existing CI scripts based on Docker
if [ "$RUN_TESTS" = "true" ]; then
    ${REPO_ROOT}/dev-tools/docker/docker_entrypoint.sh --test
else
    ${REPO_ROOT}/dev-tools/docker/docker_entrypoint.sh
fi

buildkite-agent artifact upload "build/distributions/*"
