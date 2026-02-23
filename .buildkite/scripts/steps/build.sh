#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Build step: compiles libraries, installs, strips, packages, builds test
# executables, and uploads a test bundle artifact for the separate test step.
#
# Supports all platforms:
#   - Linux x86_64: builds directly (agent runs inside Docker image)
#   - Linux aarch64: builds via Docker (docker build + docker run)
#   - macOS: builds via Gradle

set -eo pipefail

if [ "${BUILD_SNAPSHOT:=true}" != false ] ; then
    BUILD_SNAPSHOT=true
fi

if [ "$BUILD_SNAPSHOT" = false ] ; then
    export SNAPSHOT=no
else
    export SNAPSHOT=yes
fi

if [[ x"${BUILDKITE_PULL_REQUEST:-false}" != xfalse && -n "${VERSION_QUALIFIER:=""}" ]] ; then
    echo "VERSION_QUALIFIER should not be set in PR builds: was $VERSION_QUALIFIER"
    exit 2
fi

HARDWARE_ARCH=$(uname -m | sed 's/arm64/aarch64/')
OS=$(uname -s | tr "A-Z" "a-z")

# Save PATH before set_env.sh resets it (it removes buildkite-agent etc.)
ORIGINAL_PATH="$PATH"

cd "${REPO_ROOT:-.}"

if [[ "$HARDWARE_ARCH" = aarch64 && -z "${CPP_CROSS_COMPILE:-}" && "$(uname)" = Linux ]]; then
    # --- Linux aarch64 (native): Docker-based build ---
    # The Dockerfile runs docker_entrypoint.sh --build-tests which compiles
    # libraries, creates release artifacts, AND builds test executables in a
    # single docker build layer.
    BUILD_DIR="cmake-build-docker"

    docker --version

    MY_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
    TOOLS_DIR="$MY_DIR/dev-tools"

    3rd_party/pull-eigen.sh
    3rd_party/pull-valijson.sh

    . "$TOOLS_DIR/docker/prefetch_docker_image.sh"

    DOCKERFILE="$TOOLS_DIR/docker/linux_aarch64_native_builder/Dockerfile"
    TEMP_TAG=$(git rev-parse --short=14 HEAD)-linux_aarch64_native-$$

    echo "--- Building libraries and test executables (Docker)"
    prefetch_docker_base_image "$DOCKERFILE"
    docker build --no-cache --force-rm -t $TEMP_TAG --progress=plain \
        --build-arg VERSION_QUALIFIER="${VERSION_QUALIFIER:-}" \
        --build-arg SNAPSHOT=$SNAPSHOT \
        --build-arg ML_DEBUG="${ML_DEBUG:-}" \
        -f "$DOCKERFILE" .

    echo "--- Extracting build artifacts and creating test bundle"
    TEST_BUNDLE="${OS}-${HARDWARE_ARCH}-test-bundle.tar.gz"
    docker run --rm --workdir=/ml-cpp $TEMP_TAG bash -c '
        tar cf - build/distributions
    ' | tar xf -

    docker run --rm --workdir=/ml-cpp $TEMP_TAG bash -c '
        {
            find cmake-build-docker/test -name "ml_test_*" -type f -executable 2>/dev/null
            find cmake-build-docker/lib -name "*.so" 2>/dev/null
            find build/distribution -name "*.so" -not -path "*.debug*" 2>/dev/null
        } | sort -u > /tmp/bundle-files.txt
        echo "Files in bundle: $(wc -l < /tmp/bundle-files.txt)" >&2
        tar czf - -T /tmp/bundle-files.txt
    ' > "${TEST_BUNDLE}"

    docker rmi --force $TEMP_TAG

    export PATH="$ORIGINAL_PATH"

    BUNDLE_MB=$(du -m "${TEST_BUNDLE}" | cut -f1)
    echo "Test bundle: ${TEST_BUNDLE} (${BUNDLE_MB}MB)"
    buildkite-agent artifact upload "${TEST_BUNDLE}"

    if [[ "${SKIP_ARTIFACT_UPLOAD:-false}" != "true" ]] ; then
        buildkite-agent artifact upload "build/distributions/*.zip"
    fi

elif [[ "$(uname)" = "Linux" ]]; then
    # --- Linux x86_64: direct build (agent already inside Docker) ---
    BUILD_DIR="cmake-build-docker"
    dev-tools/docker/docker_entrypoint.sh --build-tests

    export PATH="$ORIGINAL_PATH"

    echo "--- Creating test bundle"
    TEST_BUNDLE="${OS}-${HARDWARE_ARCH}-test-bundle.tar.gz"

    {
        find ${BUILD_DIR}/test -name "ml_test_*" -type f \( -perm -u=x -o -name "*.exe" \) 2>/dev/null
        find ${BUILD_DIR}/lib -name "*.so" -o -name "*.dylib" 2>/dev/null
        if [ -d "build/distribution" ]; then
            find build/distribution -type f \( -name "*.so" -o -name "*.dylib" \) -not -path "*.dSYM*" 2>/dev/null
        fi
    } | sort -u > /tmp/test-bundle-files.txt

    BUNDLE_FILES=$(wc -l < /tmp/test-bundle-files.txt | tr -d ' ')
    echo "Bundling ${BUNDLE_FILES} files into ${TEST_BUNDLE}"
    tar czf "${TEST_BUNDLE}" -T /tmp/test-bundle-files.txt

    BUNDLE_MB=$(du -m "${TEST_BUNDLE}" | cut -f1)
    echo "Test bundle: ${TEST_BUNDLE} (${BUNDLE_MB}MB)"
    buildkite-agent artifact upload "${TEST_BUNDLE}"

    if [[ "${SKIP_ARTIFACT_UPLOAD:-false}" != "true" ]] ; then
        buildkite-agent artifact upload "build/distributions/*.zip"
    fi

else
    # --- macOS ---
    BUILD_DIR="cmake-build-relwithdebinfo"
    ./gradlew --info \
        -Dbuild.version_qualifier=${VERSION_QUALIFIER:-} \
        -Dbuild.snapshot=$BUILD_SNAPSHOT \
        -Dbuild.ml_debug=${ML_DEBUG:-0} \
        clean compile strip buildZip buildZipSymbols

    echo "--- Building test executables"
    . ./set_env.sh
    cmake --build ${BUILD_DIR} -j$(sysctl -n hw.logicalcpu) -t build_tests

    export PATH="$ORIGINAL_PATH"

    echo "--- Creating test bundle"
    TEST_BUNDLE="${OS}-${HARDWARE_ARCH}-test-bundle.tar.gz"

    {
        find ${BUILD_DIR}/test -name "ml_test_*" -type f \( -perm -u=x -o -name "*.exe" \) 2>/dev/null
        find ${BUILD_DIR}/lib -name "*.so" -o -name "*.dylib" 2>/dev/null
        if [ -d "build/distribution" ]; then
            find build/distribution -type f \( -name "*.so" -o -name "*.dylib" \) -not -path "*.dSYM*" 2>/dev/null
        fi
    } | sort -u > /tmp/test-bundle-files.txt

    BUNDLE_FILES=$(wc -l < /tmp/test-bundle-files.txt | tr -d ' ')
    echo "Bundling ${BUNDLE_FILES} files into ${TEST_BUNDLE}"
    tar czf "${TEST_BUNDLE}" -T /tmp/test-bundle-files.txt

    BUNDLE_MB=$(du -m "${TEST_BUNDLE}" | cut -f1)
    echo "Test bundle: ${TEST_BUNDLE} (${BUNDLE_MB}MB)"
    buildkite-agent artifact upload "${TEST_BUNDLE}"

    if [[ "${SKIP_ARTIFACT_UPLOAD:-false}" != "true" ]] ; then
        buildkite-agent artifact upload "build/distributions/*.zip"
    fi
fi
