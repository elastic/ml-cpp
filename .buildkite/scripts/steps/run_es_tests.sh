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
if [ "$BUILD_SNAPSHOT" = "yes" ] ; then
    VERSION=${VERSION}-SNAPSHOT
fi
export VERSION

export PR_AUTHOR=$(expr "$BUILDKITE_BRANCH" : '\(.*\):.*')
export PR_SOURCE_BRANCH=$(expr "$BUILDKITE_BRANCH" : '.*:\(.*\)')
export PR_TARGET_BRANCH=${BUILDKITE_PULL_REQUEST_BASE_BRANCH}

echo "id = $(id)"
ls -ld ${IVY_REPO}
sudo chown $id -R ..
mkdir -p "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION"
ls -ld ${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION
cp "${REPO_ROOT}/build/distributions/ml-cpp-$VERSION-linux-$HARDWARE_ARCH.zip" "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION/ml-cpp-$VERSION.zip"
# Since this is all local, for simplicity, cheat with the dependencies/no-dependencies split
cp "${REPO_ROOT}/build/distributions/ml-cpp-$VERSION-linux-$HARDWARE_ARCH.zip" "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION/ml-cpp-$VERSION-nodeps.zip"
# We're cheating here - the dependencies are really in the "no dependencies" zip for this flow
cp ${REPO_ROOT}/dev-tools/minimal.zip "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION/ml-cpp-$VERSION-deps.zip"
${REPO_ROOT}/dev-tools/run_es_tests.sh "${REPO_ROOT}/.." "$(cd "${IVY_REPO}" && pwd)"

