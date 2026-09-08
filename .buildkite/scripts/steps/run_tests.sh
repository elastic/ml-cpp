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
elif [[ "${ML_DEBUG:-0}" != "0" ]]; then
    BUILD_DIR="cmake-build-debug"
else
    BUILD_DIR="cmake-build-relwithdebinfo"
fi

cd "${REPO_ROOT:-.}"

echo "--- Downloading test bundle"
if [ -n "${BUILD_STEP_KEY:-}" ]; then
    buildkite-agent artifact download "${TEST_BUNDLE}" . --step "${BUILD_STEP_KEY}"
else
    buildkite-agent artifact download "${TEST_BUNDLE}" .
fi

echo "--- Extracting test bundle"
tar xzf "${TEST_BUNDLE}"
BUNDLE_MB=$(du -m "${TEST_BUNDLE}" | cut -f1)
echo "Extracted ${TEST_BUNDLE} (${BUNDLE_MB}MB)"
rm -f "${TEST_BUNDLE}"

TEST_OUTCOME=0

if [[ "$HARDWARE_ARCH" = aarch64 && -z "${CPP_CROSS_COMPILE:-}" && "$(uname)" = Linux ]]; then
    # --- Linux aarch64: run tests inside Docker container from base image ---
    BASE_IMAGE="docker.elastic.co/ml-dev/ml-linux-aarch64-native-build:17"

. ./dev-tools/docker/prefetch_docker_image.sh
    prefetch_docker_image "$BASE_IMAGE"

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
        ${TEST_TIMEOUT:+-e TEST_TIMEOUT="${TEST_TIMEOUT}"} \
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
    #
    # The container's sandbox tests are not load-bearing for coverage: both modes
    # are pinned on the host below, so whichever mode the container happens to
    # select is a bonus. Coverage therefore does not depend on Docker's default
    # seccomp profile continuing to deny unshare(CLONE_NEWUSER).

    # Both coverage modes run on the host, each pinned, so neither can quietly
    # stop executing. Measured by diagnose_userns.sh on this agent
    # (core-almalinux-8-aarch64, kernel 4.18):
    #
    #   host                             all stages OK
    #   docker (default)                 denied at unshare(CLONE_NEWUSER)
    #   docker + seccomp=unconfined      denied at mount(proc), masked /proc paths
    #   docker + seccomp + systempaths   all stages OK
    #   docker --privileged              all stages OK
    #
    # The host needs no privilege escalation for the enforced half. Runs against
    # the bundled gcc133 sysroot so the binary does not resolve against
    # AlmaLinux 8 /lib64.
    if [[ $TEST_OUTCOME -eq 0 ]]; then
        REPO_ROOT_ABS="$(pwd)"
        SYSROOT="$(pwd)/${BUILD_DIR}/lib/sysroot"
        LIB_DIRS=$(find "$(pwd)/${BUILD_DIR}/lib" "$(pwd)/build/distribution" \
            \( -name "*.so" -o -name "*.so.*" \) \
            -exec dirname {} \; 2>/dev/null | sort -u | tr '\n' ':')
        SANDBOX_TEST_DIR="${REPO_ROOT_ABS}/${BUILD_DIR}/test/lib/sandbox/unittest"
        # CPP_SRC_HOME must be set: CResourceLocator::cppRootDir() otherwise
        # falls back to "../../.." on the assumption that the cwd is a source
        # unittest directory, whereas this runs from the build tree - so the
        # spawn test looked for pytorch_inference under cmake-build-docker/ and
        # did not find it. set_env.sh exports it inside the container; nothing
        # does on the host.
        export CPP_SRC_HOME="${REPO_ROOT_ABS}"
        export LD_LIBRARY_PATH="${SYSROOT}:${LIB_DIRS}"

        echo "--- Re-running sandbox unit tests on host (enforced)"
        # Hardcoded, not overridable: this re-run is the enforced-coverage gate.
        # An env-var default (ML_SANDBOX2_HOST_REQUIRE:-enforced) would let a
        # pipeline downgrade it to fail_closed and pass required CI without ever
        # exercising a real sandbox. If a future agent genuinely cannot run
        # enforced on the host, diagnose_userns.sh reports which stage it denies.
        (cd "${SANDBOX_TEST_DIR}" && \
            ML_SANDBOX2_REQUIRE=enforced \
            ./ml_test_sandbox) || TEST_OUTCOME=$?

        # Fail-closed coverage - spawn refusal plus the kill-switch hint - is the
        # security property that keeps an unsandboxed pytorch_inference from ever
        # starting, so it must not depend on an external default either.
        #
        # user.max_user_namespaces is per-user-namespace, so setting it to 0
        # inside a namespace we own denies every further user namespace in that
        # subtree. That covers clone(CLONE_NEWUSER), which is what Sandbox2 uses,
        # as well as unshare(CLONE_NEWUSER), which is what the test's probe uses;
        # both then fail with ENOSPC. A seccomp profile denying only unshare(2)
        # would desynchronise the two and make the probe report a capability
        # Sandbox2 does not have. Verified on kernels 4.18 (this agent) and 7.0.
        if [[ $TEST_OUTCOME -eq 0 ]]; then
            echo "--- Re-running sandbox unit tests on host (fail_closed)"
            unshare --user --map-root-user sh -c "
                echo 0 > /proc/sys/user/max_user_namespaces || exit 1
                cd '${SANDBOX_TEST_DIR}' || exit 1
                ML_SANDBOX2_REQUIRE=fail_closed exec ./ml_test_sandbox
            " || TEST_OUTCOME=$?
        fi
    fi

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

    # Linux x86_64 PR agents are Buildkite k8s pods: user namespaces often work
    # but Sandbox2's mount("proc", "/proc", "proc", ...) returns EPERM, so the
    # sandbox tests self-select fail-closed coverage here. Nothing is pinned -
    # these pods' namespace support is a property of the k8s runtime, not of this
    # repository. macOS has no CSandboxedProcessSpawnerTest_Linux.cc at all.

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
    ARCHIVE_MB=$(du -m "${TEST_RESULTS_ARCHIVE}" | cut -f1)
    echo "Uploading ${TEST_RESULTS_ARCHIVE} (${ARCHIVE_MB}MB)"
    buildkite-agent artifact upload "${TEST_RESULTS_ARCHIVE}" 2>/dev/null || true
else
    echo "No test results archive created"
fi

exit $TEST_OUTCOME
