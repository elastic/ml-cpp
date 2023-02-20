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
    depends_on:
    command:
        # Download artifacts from a previous build (TODO remove build specifier once integrated with branch pipeline),
        # extract each, combine to 'uber' zip file, and upload to BuildKite's artifact store.
        - buildkite-agent artifact download "build/distributions/*.zip" . --build 0186510b-59c7-4b8c-b4f0-1da29c436ba3 --step build_test_Windows-x86_64-RelWithDebInfo
        - "./.buildkite/scripts/steps/create_dra.sh"
    agents:
      cpu: "2"
      ephemeralStorage: "20G"
      memory: "4G"
      image: "docker.elastic.co/ml-dev/ml_cpp_linux_x86_64_jdk17:3"
      # Run as a non-root user
      imageUID: "1000"
EOL
