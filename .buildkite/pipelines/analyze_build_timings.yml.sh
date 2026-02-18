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
  - label: "Analyse build timings :chart_with_upwards_trend:"
    key: "analyze_build_timings"
    command:
        - "python3 .buildkite/scripts/steps/analyze_build_timings.py"
    depends_on:
        - "build_test_linux-aarch64-RelWithDebInfo"
        - "build_test_linux-x86_64-RelWithDebInfo"
        - "build_test_macos-aarch64-RelWithDebInfo"
        - "build_test_Windows-x86_64-RelWithDebInfo"
    allow_dependency_failure: true
    soft_fail: true
    agents:
      image: "python:3-slim"
EOL
