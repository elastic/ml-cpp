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

export CMAKE_QUIET="yes"

# Default to a snapshot build
if [ -z "$BUILD_SNAPSHOT" ] ; then
    BUILD_SNAPSHOT=true
fi

# Default to running tests
if [ -z "$RUN_TESTS" ] ; then
    RUN_TESTS=true
fi

if [ "$BUILD_SNAPSHOT" = false ] ; then
    export SNAPSHOT=no
else
    export SNAPSHOT=yes
fi

# Version qualifier shouldn't be used in PR builds
if [[ -n "$BUILDKITE_PULL_REQUEST" && -n "$VERSION_QUALIFIER" ]] ; then
    echo "VERSION_QUALIFIER should not be set in PR builds: was $VERSION_QUALIFIER"
    exit 2
fi

# Tests must be run in PR builds
if [[ -n "$BUILDKITE_PULL_REQUEST" && "$RUN_TESTS" = false ]] ; then
    echo "RUN_TESTS should not be false PR builds"
    exit 3
fi

echo "environment variables:"
env
# For now, re-use our existing CI scripts based on Docker
if [ "$RUN_TESTS" = "true" ]; then
    ${REPO_ROOT}/dev-tools/docker/docker_entrypoint.sh --test
else
    ${REPO_ROOT}/dev-tools/docker/docker_entrypoint.sh
fi

buildkite-agent artifact upload "build/distributions/*"
