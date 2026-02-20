#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Test step: downloads the test bundle from the build step, extracts it, and
# runs all test suites in parallel via CTest.
#
# Used for Linux x86_64 and macOS. Linux aarch64 continues to use the
# monolithic build_and_test.sh because its Docker-based workflow makes
# splitting more complex.

set -eo pipefail

HARDWARE_ARCH=$(uname -m | sed 's/arm64/aarch64/')
OS=$(uname -s | tr "A-Z" "a-z")
TEST_BUNDLE="${OS}-${HARDWARE_ARCH}-test-bundle.tar.gz"

if [[ "$(uname)" = "Linux" ]]; then
    BUILD_DIR="cmake-build-docker"
else
    BUILD_DIR="cmake-build-relwithdebinfo"
fi

echo "--- Downloading test bundle"
buildkite-agent artifact download "${TEST_BUNDLE}" .

echo "--- Extracting test bundle"
tar xzf "${TEST_BUNDLE}"
BUNDLE_MB=$(du -m "${TEST_BUNDLE}" | cut -f1)
echo "Extracted ${TEST_BUNDLE} (${BUNDLE_MB}MB)"
rm -f "${TEST_BUNDLE}"

# Ensure test executables are executable (tar should preserve this, but be safe)
find ${BUILD_DIR}/test -name "ml_test_*" -type f -exec chmod +x {} \;

echo "--- Running tests"
TEST_OUTCOME=0

# Set LD_LIBRARY_PATH for Linux so test executables can find our shared libs
if [[ "$(uname)" = "Linux" ]]; then
    DIST_LIB="$(pwd)/build/distribution/platform/linux-${HARDWARE_ARCH}/lib"
    BUILD_LIB="$(find $(pwd)/${BUILD_DIR}/lib -name 'libMl*.so' -printf '%h\n' 2>/dev/null | sort -u | tr '\n' ':')"
    export LD_LIBRARY_PATH="${DIST_LIB}:${BUILD_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

cmake \
    -DSOURCE_DIR="$(pwd)" \
    -DBUILD_DIR="$(pwd)/${BUILD_DIR}" \
    -P cmake/run-all-tests-parallel.cmake || TEST_OUTCOME=$?

# Upload test results
echo "--- Uploading test results"
TEST_RESULTS_ARCHIVE=${OS}-${HARDWARE_ARCH}-unit_test_results.tgz
find . \( -path "*/**/ml_test_*.out" -o -path "*/**/*.junit" \) -print0 | tar czf ${TEST_RESULTS_ARCHIVE} --null -T - 2>/dev/null || true
if [ -f "${TEST_RESULTS_ARCHIVE}" ]; then
    buildkite-agent artifact upload "${TEST_RESULTS_ARCHIVE}" 2>/dev/null || true
fi

exit $TEST_OUTCOME
