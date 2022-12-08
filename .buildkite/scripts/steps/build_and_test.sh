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
    ${REPO_ROOT}/dev-tools/docker/docker_entrypoint.sh --test
else
    ${REPO_ROOT}/dev-tools/docker/docker_entrypoint.sh
fi

buildkite-agent artifact upload "build/distributions/*"
