#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

cat <<EOL
steps:
  - label: "Ingest build timings :elasticsearch:"
    key: "ingest_build_timings"
    command:
        - "pip3 install --quiet requests 2>/dev/null || true"
        - "python3 dev-tools/ingest_build_timings.py --pipeline \$BUILDKITE_PIPELINE_SLUG --build \$BUILDKITE_BUILD_NUMBER"
    depends_on:
        - "build_test_linux-aarch64-RelWithDebInfo"
        - "build_test_linux-x86_64-RelWithDebInfo"
        - "build_test_macos-aarch64-RelWithDebInfo"
        - "build_test_Windows-x86_64-RelWithDebInfo"
    allow_dependency_failure: true
    soft_fail: true
    agents:
      image: "python:3-slim"
    env:
      BUILDKITE_API_READ_TOKEN: ""
EOL
