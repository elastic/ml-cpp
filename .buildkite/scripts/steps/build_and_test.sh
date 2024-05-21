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
if [ "${BUILD_SNAPSHOT:=true}" != false ] ; then
    BUILD_SNAPSHOT=true
fi

# Default to running tests.
# Not every build step will be able to run the tests
# e.g. cross compilations.
if [ -z "$RUN_TESTS" ] ; then
    RUN_TESTS=true
fi

if [ "$BUILD_SNAPSHOT" = false ] ; then
    export SNAPSHOT=no
else
    export SNAPSHOT=yes
fi

# Version qualifier shouldn't be used in PR builds
if [[ x"$BUILDKITE_PULL_REQUEST" != xfalse && -n "${VERSION_QUALIFIER:=""}" ]] ; then
    echo "VERSION_QUALIFIER should not be set in PR builds: was $VERSION_QUALIFIER"
    exit 2
fi

VERSION=$(cat ${REPO_ROOT}/gradle.properties | grep '^elasticsearchVersion' | awk -F= '{ print $2 }' | xargs echo)
HARDWARE_ARCH=$(uname -m | sed 's/arm64/aarch64/')

TEST_OUTCOME=0
if [[ `uname` = Linux && -z "$CPP_CROSS_COMPILE" ]] ; then
  # On native Linux build using Docker
  # This means that we can tolerate a very old Git version inside the Docker container

  # The Docker version is helpful to identify version-specific Docker bugs
  docker --version

  KERNEL_VERSION=`uname -r`
  GLIBC_VERSION=`ldconfig --version | head -1 | sed 's/ldconfig//'`

  if [ "$HARDWARE_ARCH" = aarch64 ] ; then
      DOCKER_BUILD_ARG=linux_aarch64_native
  else
      DOCKER_BUILD_ARG=linux
  fi

  if [ "$RUN_TESTS" = false ] ; then
    ${REPO_ROOT}/dev-tools/docker_build.sh $DOCKER_BUILD_ARG
  else
    ${REPO_ROOT}/dev-tools/docker_test.sh --extract-unit-tests $DOCKER_BUILD_ARG || TEST_OUTCOME=$?
  fi
fi

# If this is a PR build then it's redundant to cross compile aarch64 (as
# we build and test aarch64 natively for PR builds) but there's a benefit
# to building one platform with debug enabled to detect code that only
# compiles with optimisation
if [[ x"$BUILDKITE_PULL_REQUEST" != xfalse && "$CPP_CROSS_COMPILE" = "aarch64" ]] ; then
    export ML_DEBUG=1
fi

if [[ `uname` = "Darwin" && "$HARDWARE_ARCH" = "aarch64" ]]; then
  # For ARM macOS, build directly on the machine
  sudo -E ${REPO_ROOT}/dev-tools/download_macos_deps.sh
  if [ "$RUN_TESTS" = false ] ; then
      TASKS="clean buildZip buildZipSymbols"
  else
      TASKS="clean buildZip buildZipSymbols check"
  fi
  # For macOS we usually only use a particular version as our build platform
  # once Xcode has stopped receiving updates for it. However, with Big Sur
  # on ARM we couldn't do this, as Big Sur was the first macOS version for
  # ARM. Therefore, the compiler may get upgraded on a CI server, and we
  # need to hardcode the version that was used to build Boost for that
  # version of Elasticsearch.
  export BOOSTCLANGVER=120

  (cd ${REPO_ROOT} && ./gradlew --info -Dbuild.version_qualifier=${VERSION_QUALIFIER:-} -Dbuild.snapshot=${BUILD_SNAPSHOT:-} -Dbuild.ml_debug=${ML_DEBUG:-} $TASKS) || TEST_OUTCOME=$?

# If cross-compiling re-use our existing CI scripts based on Docker.
elif [ -n "$CPP_CROSS_COMPILE" ] ; then
  if [ "$RUN_TESTS" = "true" ]; then
    ${REPO_ROOT}/dev-tools/docker/docker_entrypoint.sh --test
    grep passed build/test_status.txt || TEST_OUTCOME=$?
  else
    ${REPO_ROOT}/dev-tools/docker/docker_entrypoint.sh
  fi
fi

if ! [[ "$HARDWARE_ARCH" = aarch64 && -n "$CPP_CROSS_COMPILE" ]] && [[ $TEST_OUTCOME -eq 0 ]] ; then
  buildkite-agent artifact upload "build/distributions/*.zip"
fi

if [[ -z "$CPP_CROSS_COMPILE" ]] ; then
  OS=$(uname -s | tr "A-Z" "a-z")
  TEST_RESULTS_ARCHIVE=${OS}-${HARDWARE_ARCH}-unit_test_results.tgz
  find . -path  "*/**/ml_test_*.out" -o -path "*/**/*.junit" | xargs tar cvzf ${TEST_RESULTS_ARCHIVE}
  buildkite-agent artifact upload "${TEST_RESULTS_ARCHIVE}"
fi

exit $TEST_OUTCOME
