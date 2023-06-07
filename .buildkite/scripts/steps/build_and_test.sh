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
if [[ "$HARDWARE_ARCH" = aarch64 && -z "$CPP_CROSS_COMPILE" && `uname` = Linux ]] ; then 
  # On Linux native aarch64 build using Docker
  
  # The Docker version is helpful to identify version-specific Docker bugs
  docker --version
  
  KERNEL_VERSION=`uname -r`
  GLIBC_VERSION=`ldconfig --version | head -1 | sed 's/ldconfig//'`

  if [ "$RUN_TESTS" = false ] ; then
    ${REPO_ROOT}/dev-tools/docker_build.sh linux_aarch64_native
  else
    ${REPO_ROOT}/dev-tools/docker_test.sh --extract-unit-tests linux_aarch64_native || TEST_OUTCOME=$?
    if [[ $TEST_OUTCOME -eq 0 ]]; then
      echo "Re-running seccomp unit tests outside of Docker container - kernel: $KERNEL_VERSION glibc: $GLIBC_VERSION"
      (cd ${REPO_ROOT}/cmake-build-docker/test/lib/seccomp/unittest && \
        LD_LIBRARY_PATH=`cd ../../../../../build/distribution/platform/linux-aarch64/lib && pwd` ./ml_test_seccomp) || TEST_OUTCOME=$?
    fi
  fi
fi

# If this is a PR build then it's redundant to cross compile aarch64 (as
# we build and test aarch64 natively for PR builds) but there's a benefit
# to building one platform with debug enabled to detect code that only
# compiles with optimisation
if [[ x"$BUILDKITE_PULL_REQUEST" != xfalse && "$CPP_CROSS_COMPILE" = "aarch64" ]] ; then
    export ML_DEBUG=1
fi

# For now, re-use our existing CI scripts based on Docker
# Don't perform these steps for native linux aarch64 builds as
# they are built using docker, see above.
if ! [[ "$HARDWARE_ARCH" = aarch64 && -z "$CPP_CROSS_COMPILE" ]] ; then 
  if [ "$RUN_TESTS" = "true" ]; then
    ${REPO_ROOT}/dev-tools/docker/docker_entrypoint.sh --test
    grep passed build/test_status.txt || TEST_OUTCOME=$?
  else
    ${REPO_ROOT}/dev-tools/docker/docker_entrypoint.sh
  fi
else
  if [[ `uname` = "Darwin" && "$HARDWARE_ARCH" = "aarch64" ]]; then
     # For macOS, build directly on the machine
     ${REPO_ROOT}/dev-tools/download_macos_deps.sh
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
     if [ "$HARDWARE_ARCH" = aarch64 ] ; then
         export BOOSTCLANGVER=13
     fi

     (cd ${REPO_ROOT} && ./gradlew --info -Dbuild.version_qualifier=${VERSION_QUALIFIER:-} -Dbuild.snapshot=$BUILD_SNAPSHOT -Dbuild.ml_debug=$ML_DEBUG $TASKS) || TEST_OUTCOME=$?
  fi
fi

# We don't upload artifacts from the cross compiled aarch64 build as it has been built with full debug
# and assertions enabled
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
