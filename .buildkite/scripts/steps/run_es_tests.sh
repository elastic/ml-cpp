#!/bin/bash

echo "pwd = $(pwd)"
ls -lR

HARDWARE_ARCH=$(uname -m | sed 's/arm64/aarch64/')

VERSION=$(cat ${REPO_ROOT}/gradle.properties | grep '^elasticsearchVersion' | awk -F= '{ print $2 }' | xargs echo)
if [ "$BUILD_SNAPSHOT" = "yes" ] ; then
    VERSION=${VERSION}-SNAPSHOT
fi

PR_AUTHOR=$(expr "$BUILDKITE_BRANCH" : '\(.*\):.*')
PR_SOURCE_BRANCH=$(expr "$BUILDKITE_BRANCH" : '.*:\(.*\)')
PR_TARGET_BRANCH=${BUILDKITE_PULL_REQUEST_BASE_BRANCH}

mkdir -p "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION"
cp "${REPO_ROOT}/build/distributions/ml-cpp-$VERSION-linux-$HARDWARE_ARCH.zip" "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION/ml-cpp-$VERSION.zip"
# Since this is all local, for simplicity, cheat with the dependencies/no-dependencies split
cp "${REPO_ROOT}/build/distributions/ml-cpp-$VERSION-linux-$HARDWARE_ARCH.zip" "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION/ml-cpp-$VERSION-nodeps.zip"
# We're cheating here - the dependencies are really in the "no dependencies" zip for this flow
cp ${REPO_ROOT}/dev-tools/minimal.zip "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION/ml-cpp-$VERSION-deps.zip"
${REPO_ROOT}/dev-tools/run_es_tests.sh "${REPO_ROOT}/.." "$(cd "${IVY_REPO}" && pwd)"

