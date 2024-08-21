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
  - label: "Create DRA artifacts"
    key: "create_dra_artifacts"
    command:
        - "./.buildkite/scripts/steps/create_dra.sh"
    depends_on:
        - "build_test_linux-aarch64-RelWithDebInfo"
        - "build_test_linux-x86_64-RelWithDebInfo"
        - "build_test_macos-x86_64-RelWithDebInfo"
        - "build_test_macos-aarch64-RelWithDebInfo"
        - "build_test_Windows-x86_64-RelWithDebInfo"

    agents:
      cpu: "2"
      ephemeralStorage: "20G"
      memory: "4G"
      image: "docker.elastic.co/ml-dev/ml_cpp_linux_x86_64_jdk17:3"
      # Run as a non-root user
      imageUID: "1000"
EOL
