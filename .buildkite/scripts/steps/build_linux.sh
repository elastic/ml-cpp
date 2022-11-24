#!/bin/bash

KERNEL_VERSION=$(uname -r)
GLIBC_VERSION=$(ldconfig --version | head -1 | sed 's/ldconfig//')

# Build and test the native Linux architecture
if [ "$BUILD_ARCH" = x86_64 ] ; then

    if [ "$RUN_TESTS" = false ] ; then
        ./.buildkite/scripts/steps/build_and_test.sh linux
    else
        ./.buildkite/scripts/steps/build_and_test.sh --extract-unit-tests linux
        #echo "Re-running seccomp unit tests outside of Docker container - kernel: $KERNEL_VERSION glibc: $GLIBC_VERSION"
        #(cd ../cmake-build-docker/test/lib/seccomp/unittest && LD_LIBRARY_PATH=`cd ../../../../../build/distribution/platform/linux-x86_64/lib && pwd` ./ml_test_seccomp)
    fi

elif [ "$BUILD_ARCH" = aarch64 ] ; then
    if [ "$RUN_TESTS" = false ] ; then
        ./.buildkite/scripts/steps/build_and_test.sh linux_aarch64_native
    else
        ./.buildkite/scripts/steps/build_and_test.sh --extract-unit-tests linux_aarch64_native
        echo "Re-running seccomp unit tests outside of Docker container - kernel: $KERNEL_VERSION glibc: $GLIBC_VERSION"
        (cd ../cmake-build-docker/test/lib/seccomp/unittest && LD_LIBRARY_PATH=`cd ../../../../../build/distribution/platform/linux-aarch64/lib && pwd` ./ml_test_seccomp)
    fi
fi

## If this is a PR build then run some Java integration tests
#if [ -n "$PR_AUTHOR" ] ; then
#    IVY_REPO="${GIT_TOPLEVEL}/../ivy"
#    mkdir -p "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION"
#    cp "../build/distributions/ml-cpp-$VERSION-linux-$BUILD_ARCH.zip" "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION/ml-cpp-$VERSION.zip"
#    # Since this is all local, for simplicity, cheat with the dependencies/no-dependencies split
#    cp "../build/distributions/ml-cpp-$VERSION-linux-$BUILD_ARCH.zip" "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION/ml-cpp-$VERSION-nodeps.zip"
#    # We're cheating here - the dependencies are really in the "no dependencies" zip for this flow
#    cp minimal.zip "${IVY_REPO}/maven/org/elasticsearch/ml/ml-cpp/$VERSION/ml-cpp-$VERSION-deps.zip"
#    ./run_es_tests.sh "${GIT_TOPLEVEL}/.." "$(cd "${IVY_REPO}" && pwd)"
#fi

