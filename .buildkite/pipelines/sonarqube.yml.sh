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
  - label: "Generate compile_commands.json"
    key: "export_compile_commands"
    # depends_on: "check_style"
    agents:
      image: "docker.elastic.co/ml-dev/ml-linux-build:30"
    env:
      - PATH: "/usr/local/gcc103/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
      - VAULT_SONAR_TOKEN_PATH: "secret/ci/elastic-ml-cpp/sonar-analyze-token"
    command: ".buildkite/scripts/steps/run_sonar-scanner.sh"
    # artifact_paths:
    #   - "cmake-build-docker/compile_commands.json"
    notify:
      - github_commit_status:
          context: "Generate compile_commands.json"
  # - label: "Static code analysis with SonarQube :sonarqube:"
  #   key: "sonarqube"
  #   depends_on: "export_compile_commands"
  #   env:
  #     VAULT_SONAR_TOKEN_PATH: "secret/ci/elastic-ml-cpp/sonar-analyze-token"
  #   command: 
  #     - "buildkite-agent artifact download cmake-build-docker/compile_commands.json ."
  #     - ".buildkite/scripts/steps/update_compile_commands.sh"
  #     - "/scan-source-code.sh"
  #   agents:
  #     image: "docker.elastic.co/cloud-ci/sonarqube/buildkite-scanner:latest"
  #   notify:
  #     - github_commit_status:
  #         context: "Static code analysis with SonarQube"
EOL