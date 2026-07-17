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

echo "pwd = $(pwd)"

export HARDWARE_ARCH=$(uname -m | sed 's/arm64/aarch64/')

VERSION=$(cat ${REPO_ROOT}/gradle.properties | grep '^elasticsearchVersion' | awk -F= '{ print $2 }' | xargs echo)
if [ "${BUILD_SNAPSHOT:=true}" = "true" ] ; then
    VERSION=${VERSION}-SNAPSHOT
fi
export VERSION

export PR_AUTHOR=$(expr "$BUILDKITE_BRANCH" : '\(.*\):.*')
export PR_SOURCE_BRANCH=$(expr "$BUILDKITE_BRANCH" : '.*:\(.*\)')
export PR_TARGET_BRANCH=${BUILDKITE_PULL_REQUEST_BASE_BRANCH}

# Set up GCS credentials for Gradle build cache persistence (if available).
# The post-checkout hook writes the GCS service account key for sccache;
# reuse the same credentials for the Gradle cache bucket.
if [ -n "${SCCACHE_GCS_BUCKET:-}" ] && [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    export GRADLE_BUILD_CACHE_GCS_BUCKET="${SCCACHE_GCS_BUCKET}"
    # Install gsutil if not already present
    if ! command -v gsutil &>/dev/null; then
        echo "--- Installing gsutil"
        pip3 install --quiet gsutil 2>/dev/null || pip install --quiet gsutil 2>/dev/null || echo "Warning: failed to install gsutil"
    fi
fi

mkdir -p "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION"
cp "build/distributions/ml-cpp-$VERSION-linux-$HARDWARE_ARCH.zip" "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION/ml-cpp-$VERSION.zip"
# Since this is all local, for simplicity, cheat with the dependencies/no-dependencies split
cp "build/distributions/ml-cpp-$VERSION-linux-$HARDWARE_ARCH.zip" "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION/ml-cpp-$VERSION-nodeps.zip"
# We're cheating here - the dependencies are really in the "no dependencies" zip for this flow
cp dev-tools/minimal.zip "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION/ml-cpp-$VERSION-deps.zip"
./dev-tools/run_es_tests.sh ".." "$(cd "${IVY_REPO}" && pwd)"

