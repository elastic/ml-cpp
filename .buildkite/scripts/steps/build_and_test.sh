#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

echo "environment variables:"
env
# For now, re-use our existing CI scripts based on Docker
if [ "$RUN_TESTS" = "true" ]; then
    buildkite-agent artifact download "build/*" .
    buildkite-agent artifact download "cmake-build-docker/*" .
    #${REPO_ROOT}/dev-tools/docker/docker_entrypoint.sh --test
    # Convert any failure of this make command into the word passed or failed in
    # a status file - this allows the Docker image build to succeed if the only
    # failure is the unit tests, and then the detailed test results can be
    # copied from the image
    echo passed > build/test_status.txt
    cmake --build cmake-build-docker -v -j`nproc` -t test || echo failed > build/test_status.txt
else
    ${REPO_ROOT}/dev-tools/docker/docker_entrypoint.sh
    buildkite-agent artifact upload "build/distributions/*"
    buildkite-agent artifact upload "cmake-build-docker/**/*"
fi
