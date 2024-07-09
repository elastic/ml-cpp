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
  - label: "Static code analysis with SonarQube :sonarqube:"
    key: "sonarqube"
    depends_on: "build_test_macos-aarch64-RelWithDebInfo"
    env:
      VAULT_SONAR_TOKEN_PATH: "secret/ci/elastic-ml-cpp/sonar-analyze-token"
    command: 
      - "buildkite-agent artifact download cmake-build-docker/compile_commands.json ."
      - ".buildkite/scripts/steps/sonar-qube.sh"
    agents:
      image: "docker.elastic.co/cloud-ci/sonarqube/buildkite-scanner:latest"
    notify:
      - github_commit_status:
          context: "Static code analysis with SonarQube"
EOL