#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#

# This script gets run within the Docker container when a build is done in a
# Docker container.
#
# It is not intended to be run outside of a Docker container (although it
# should work if it is).

set -e

if [ -z "$CMAKE_QUIET" ]; then
  CMAKE_VERBOSE="-v"
fi

# Change directory to the root of the Git repository
MY_DIR=`dirname "$BASH_SOURCE"`
cd "$MY_DIR/../.."

# Set a consistent environment
. ./set_env.sh

# Detect actual CPU count, respecting cgroup limits (Docker/k8s).
# nproc may report host CPUs rather than the container's allocation.
detect_cpus() {
    local cgroup_cpus=""
    # Cgroup v2
    if [ -f /sys/fs/cgroup/cpu.max ]; then
        local quota period
        read quota period < /sys/fs/cgroup/cpu.max
        if [ "$quota" != "max" ] && [ "$period" -gt 0 ] 2>/dev/null; then
            cgroup_cpus=$(( (quota + period - 1) / period ))
        fi
    # Cgroup v1
    elif [ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
        local quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        local period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        if [ "$quota" -gt 0 ] && [ "$period" -gt 0 ] 2>/dev/null; then
            cgroup_cpus=$(( (quota + period - 1) / period ))
        fi
    fi
    if [ -n "$cgroup_cpus" ] && [ "$cgroup_cpus" -gt 0 ] 2>/dev/null; then
        echo "$cgroup_cpus"
    else
        nproc
    fi
}

NCPUS=$(detect_cpus)
echo "CPU detection: nproc=$(nproc), cgroup-aware=${NCPUS}"

# Set up sccache with GCS backend if credentials are available.
# SCCACHE_GCS_BUCKET is exported by the Buildkite post-checkout hook.
if [ -n "${SCCACHE_GCS_BUCKET:-}" ]; then
  source ./dev-tools/setup_sccache.sh
fi

# Note: no need to clean due to the .dockerignore file

# Configure the build
cmake -B cmake-build-docker ${CMAKE_FLAGS}

# Build the code
cmake --build cmake-build-docker ${CMAKE_VERBOSE} -j${NCPUS} -t install

# Strip the binaries
dev-tools/strip_binaries.sh

# Get the version number
PRODUCT_VERSION=`cat "$CPP_SRC_HOME/gradle.properties" | grep '^elasticsearchVersion' | awk -F= '{ print $2 }' | xargs echo`
if [ -n "$VERSION_QUALIFIER" ] ; then
    PRODUCT_VERSION="$PRODUCT_VERSION-$VERSION_QUALIFIER"
fi
if [ "$SNAPSHOT" = yes ] ; then
    PRODUCT_VERSION="$PRODUCT_VERSION-SNAPSHOT"
fi

ARTIFACT_NAME=`cat "$CPP_SRC_HOME/gradle.properties" | grep '^artifactName' | awk -F= '{ print $2 }' | xargs echo`

# Create the output artifacts
cd build/distribution
mkdir -p ../distributions
ZIP_LEVEL=${ZIP_COMPRESSION_LEVEL:-9}
echo "Zip compression level: ${ZIP_LEVEL}"
# Exclude import libraries, test support libraries, debug files and core dumps
zip -${ZIP_LEVEL} ../distributions/$ARTIFACT_NAME-$PRODUCT_VERSION-$BUNDLE_PLATFORM.zip `find * | egrep -v '\.lib$|unit_test_framework|libMlTest|\.dSYM|-debug$|\.pdb$|/core'`
# Include only debug files
zip -${ZIP_LEVEL} ../distributions/$ARTIFACT_NAME-$PRODUCT_VERSION-debug-$BUNDLE_PLATFORM.zip `find * | egrep '\.dSYM|-debug$|\.pdb$'`
cd ../..

if [ "x$1" = "x--test" ] ; then
    # Convert any failure of this make command into the word passed or failed in
    # a status file - this allows the Docker image build to succeed if the only
    # failure is the unit tests, and then the detailed test results can be
    # copied from the image
    echo passed > build/test_status.txt
    # Each test suite spawns ctest --parallel <ncpus> internally, so limit
    # the number of suites running concurrently to avoid resource contention.
    # On low-core machines (<=4), cap at ncpus-1 to leave headroom for
    # timing-sensitive tests (e.g. CKMostCorrelatedTest/testScale).
    # For higher core counts, ceil(ncpus/2) balances parallelism vs
    # contention — ceil(ncpus/3) was too conservative on 8-core machines.
    if [ "$NCPUS" -le 4 ]; then
        TEST_PARALLEL=2
    else
        TEST_PARALLEL=$(( (NCPUS + 1) / 2 ))
    fi
    echo "Test parallelism: nproc=${NCPUS}, TEST_PARALLEL=${TEST_PARALLEL} (cmake --build -j ${TEST_PARALLEL})"
    cmake --build cmake-build-docker ${CMAKE_VERBOSE} -j ${TEST_PARALLEL} -t test_individually || echo failed > build/test_status.txt
fi

# Print sccache stats if it was used
if [ -n "${SCCACHE_PATH:-}" ]; then
  "$SCCACHE_PATH" --show-stats || true
  "$SCCACHE_PATH" --stop-server || true
fi

