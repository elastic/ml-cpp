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
  - label: "Run SonarQube scanner :sonarqube:"
    key: "export_compile_commands"
    # depends_on: "build_test_linux-x86_64-RelWithDebInfo"
    soft_fail: true
    agents:
      cpu: 4
      ephemeralStorage: "60G"
      memory: "16GB"
      image: "docker.elastic.co/ml-dev/ml-linux-build:30"
    env:
      PATH: "/usr/local/gcc103/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
      VAULT_SONAR_TOKEN_PATH: "secret/ci/elastic-ml-cpp/sonar-analyze-token"
    command: 
      - ".buildkite/scripts/steps/run_sonar-scanner.sh"
    # artifact_paths:
    #   - "cmake-build-docker/compile_commands.json"
    notify:
      - github_commit_status:
          context: "Run SonarQube scanner"
EOL