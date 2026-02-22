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

# Save PATH before set_env.sh resets it (it removes buildkite-agent etc.)
ORIGINAL_PATH="$PATH"

cd "${REPO_ROOT:-.}"

if [[ "$(uname)" = "Linux" ]]; then
    BUILD_DIR="cmake-build-docker"
    dev-tools/docker/docker_entrypoint.sh

    echo "--- Building test executables"
    . ./set_env.sh
    cmake --build ${BUILD_DIR} -j$(nproc) -t build_tests
else
    BUILD_DIR="cmake-build-relwithdebinfo"
    ./gradlew --info \
        -Dbuild.version_qualifier=${VERSION_QUALIFIER:-} \
        -Dbuild.snapshot=$BUILD_SNAPSHOT \
        -Dbuild.ml_debug=${ML_DEBUG:-0} \
        clean compile strip buildZip buildZipSymbols

    echo "--- Building test executables"
    . ./set_env.sh
    cmake --build ${BUILD_DIR} -j$(sysctl -n hw.logicalcpu) -t build_tests
fi

# Restore PATH for buildkite-agent access
export PATH="$ORIGINAL_PATH"

# --- Create and upload test bundle ---
# The bundle contains test executables AND all shared libraries (ours + 3rd
# party) so the test step can run on a different agent without needing the
# original build tree. DYLD_LIBRARY_PATH / LD_LIBRARY_PATH overrides the
# absolute rpaths baked in at link time.
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
