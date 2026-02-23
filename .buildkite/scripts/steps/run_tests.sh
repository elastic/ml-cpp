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
# The test bundle contains pre-built test executables and ALL shared libraries.
# We set DYLD_LIBRARY_PATH (macOS) / LD_LIBRARY_PATH (Linux) to override the
# absolute rpaths baked in at link time, allowing the executables to find libs
# even though this agent's workspace path differs from the build agent's.

set -eo pipefail

HARDWARE_ARCH=$(uname -m | sed 's/arm64/aarch64/')
OS=$(uname -s | tr "A-Z" "a-z")
TEST_BUNDLE="${OS}-${HARDWARE_ARCH}-test-bundle.tar.gz"

if [[ "$(uname)" = "Linux" ]]; then
    BUILD_DIR="cmake-build-docker"
else
    BUILD_DIR="cmake-build-relwithdebinfo"
fi

cd "${REPO_ROOT:-.}"

echo "--- Downloading test bundle"
buildkite-agent artifact download "${TEST_BUNDLE}" .

echo "--- Extracting test bundle"
tar xzf "${TEST_BUNDLE}"
BUNDLE_MB=$(du -m "${TEST_BUNDLE}" | cut -f1)
echo "Extracted ${TEST_BUNDLE} (${BUNDLE_MB}MB)"
rm -f "${TEST_BUNDLE}"

TEST_OUTCOME=0

if [[ "$HARDWARE_ARCH" = aarch64 && -z "${CPP_CROSS_COMPILE:-}" && "$(uname)" = Linux ]]; then
    # --- Linux aarch64: run tests inside Docker container from base image ---
    BASE_IMAGE="docker.elastic.co/ml-dev/ml-linux-aarch64-native-build:17"

    echo "--- Running tests (Docker)"
    docker run --rm \
        -v "$(pwd)/${BUILD_DIR}:/ml-cpp/${BUILD_DIR}" \
        -v "$(pwd)/build:/ml-cpp/build" \
        -v "$(pwd)/lib:/ml-cpp/lib" \
        -v "$(pwd)/bin:/ml-cpp/bin" \
        -v "$(pwd)/cmake:/ml-cpp/cmake:ro" \
        -v "$(pwd)/set_env.sh:/ml-cpp/set_env.sh:ro" \
        -v "$(pwd)/gradle.properties:/ml-cpp/gradle.properties:ro" \
        -e BOOST_TEST_OUTPUT_FORMAT_FLAGS="${BOOST_TEST_OUTPUT_FORMAT_FLAGS:-}" \
        -w /ml-cpp \
        $BASE_IMAGE bash -c '
            source ./set_env.sh

            LIB_DIRS=$(find /ml-cpp/cmake-build-docker/lib /ml-cpp/build/distribution \
                -name "*.so" -exec dirname {} \; 2>/dev/null | sort -u | tr "\n" ":")
            export LD_LIBRARY_PATH="${LIB_DIRS}/usr/local/gcc133/lib64:/usr/local/gcc133/lib"

            chmod -R +x cmake-build-docker/test/ 2>/dev/null

            cmake \
                -DSOURCE_DIR=/ml-cpp \
                -DBUILD_DIR=/ml-cpp/cmake-build-docker \
                -P cmake/run-all-tests-parallel.cmake
        ' || TEST_OUTCOME=$?

    # Seccomp tests run inside the Docker container which shares the host
    # kernel, so the kernel's seccomp filters are exercised without needing
    # a separate outside-Docker run.

else
    # --- Linux x86_64 / macOS: run tests directly ---
    . ./set_env.sh

    find ${BUILD_DIR}/test -name "ml_test_*" -type f -exec chmod +x {} \;

    LIB_DIRS=$(find "$(pwd)/${BUILD_DIR}/lib" "$(pwd)/build/distribution" \
        \( -name "*.so" -o -name "*.dylib" \) -not -path "*.dSYM*" \
        -exec dirname {} \; 2>/dev/null | sort -u | tr '\n' ':')

    if [[ "$(uname)" = "Linux" ]]; then
        export LD_LIBRARY_PATH="${LIB_DIRS}/usr/local/gcc133/lib64:/usr/local/gcc133/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    else
        export DYLD_LIBRARY_PATH="${LIB_DIRS}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    fi

    echo "--- Running tests"
    cmake \
        -DSOURCE_DIR="$(pwd)" \
        -DBUILD_DIR="$(pwd)/${BUILD_DIR}" \
        -P cmake/run-all-tests-parallel.cmake || TEST_OUTCOME=$?
fi

# Upload test results
echo "--- Uploading test results"
TEST_RESULTS_ARCHIVE=${OS}-${HARDWARE_ARCH}-unit_test_results.tgz
find . \( -path "*/**/ml_test_*.out" -o -path "*/**/*.junit" \) -print0 | tar czf ${TEST_RESULTS_ARCHIVE} --null -T - 2>/dev/null || true
if [ -f "${TEST_RESULTS_ARCHIVE}" ]; then
    buildkite-agent artifact upload "${TEST_RESULTS_ARCHIVE}" 2>/dev/null || true
fi

exit $TEST_OUTCOME
