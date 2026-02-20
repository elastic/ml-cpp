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
# Used for Linux x86_64 and macOS. Linux aarch64 continues to use the
# monolithic build_and_test.sh because its Docker-based workflow makes
# splitting more complex.

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

# Ensure we're in the repo root
cd "${REPO_ROOT:-.}"

if [[ "$(uname)" = "Linux" ]]; then
    # --- Linux x86_64: build directly (k8s pod IS the build environment) ---
    BUILD_DIR="cmake-build-docker"

    # Build libraries, install, strip, package
    dev-tools/docker/docker_entrypoint.sh

    # Build test executables (docker_entrypoint.sh without --test doesn't)
    echo "--- Building test executables"
    . ./set_env.sh
    cmake --build ${BUILD_DIR} -j$(nproc) -t build_tests
else
    # --- macOS: build via Gradle ---
    BUILD_DIR="cmake-build-relwithdebinfo"
    ./gradlew --info \
        -Dbuild.version_qualifier=${VERSION_QUALIFIER:-} \
        -Dbuild.snapshot=$BUILD_SNAPSHOT \
        -Dbuild.ml_debug=${ML_DEBUG:-0} \
        clean compile strip buildZip buildZipSymbols

    # Build test executables
    echo "--- Building test executables"
    . ./set_env.sh
    cmake --build ${BUILD_DIR} -j$(sysctl -n hw.logicalcpu) -t build_tests
fi

# --- Create and upload test bundle ---
echo "--- Creating test bundle"
TEST_BUNDLE="${OS}-${HARDWARE_ARCH}-test-bundle.tar.gz"

{
    # Test executables
    find ${BUILD_DIR}/test -name "ml_test_*" -type f \( -perm -u=x -o -name "*.exe" \) 2>/dev/null
    # Our shared libraries (built by us, needed at runtime)
    find ${BUILD_DIR}/lib -name "libMl*.so" -o -name "libMl*.dylib" 2>/dev/null
    # The installed distribution (contains shared libs for LD_LIBRARY_PATH / @rpath)
    if [ -d "build/distribution" ]; then
        find build/distribution -type f \( -name "libMl*" -o -name "*.so" -o -name "*.dylib" \) 2>/dev/null
    fi
} | sort -u > /tmp/test-bundle-files.txt

BUNDLE_FILES=$(wc -l < /tmp/test-bundle-files.txt | tr -d ' ')
echo "Bundling ${BUNDLE_FILES} files into ${TEST_BUNDLE}"

tar czf "${TEST_BUNDLE}" -T /tmp/test-bundle-files.txt

BUNDLE_MB=$(du -m "${TEST_BUNDLE}" | cut -f1)
echo "Test bundle: ${TEST_BUNDLE} (${BUNDLE_MB}MB)"
buildkite-agent artifact upload "${TEST_BUNDLE}"

# --- Upload distribution artifacts ---
if [[ "${SKIP_ARTIFACT_UPLOAD:-false}" != "true" ]] ; then
    buildkite-agent artifact upload "build/distributions/*.zip"
fi
