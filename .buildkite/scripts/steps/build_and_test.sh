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

VERSION=$(cat ${REPO_ROOT}/gradle.properties | grep '^elasticsearchVersion' | awk -F= '{ print $2 }' | xargs echo)
HARDWARE_ARCH=$(uname -m | sed 's/arm64/aarch64/')

echo "environment variables:"
env

if [[ "$HARDWARE_ARCH" = aarch64 && -z "$CPP_CROSS_COMPILE" ]] ; then 
  # On Linux native aarch64 build using Docker
  
  # The Docker version is helpful to identify version-specific Docker bugs
  docker --version
  
  KERNEL_VERSION=`uname -r`
  GLIBC_VERSION=`ldconfig --version | head -1 | sed 's/ldconfig//'`

  if [ "$RUN_TESTS" = false ] ; then
    ${REPO_ROOT}/dev-tools/docker_build.sh linux_aarch64_native
  else
    ${REPO_ROOT}/dev-tools/docker_test.sh --extract-unit-tests linux_aarch64_native
    echo "Re-running seccomp unit tests outside of Docker container - kernel: $KERNEL_VERSION glibc: $GLIBC_VERSION"
    (cd ${REPO_ROOT}/cmake-build-docker/test/lib/seccomp/unittest && LD_LIBRARY_PATH=`cd ../../../../../build/distribution/platform/linux-aarch64/lib && pwd` ./ml_test_seccomp)
  fi
  exit $?
fi

# For now, re-use our existing CI scripts based on Docker
if [ "$RUN_TESTS" = "true" ]; then
    ${REPO_ROOT}/dev-tools/docker/docker_entrypoint.sh --test
else
    ${REPO_ROOT}/dev-tools/docker/docker_entrypoint.sh
fi

buildkite-agent artifact upload "build/distributions/*"
