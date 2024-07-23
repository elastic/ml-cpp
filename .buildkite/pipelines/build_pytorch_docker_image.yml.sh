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
---
steps:
  - label: "Build PyTorch Docker Image"
    key: "build_pytorch_docker_image"
    command: "./dev-tools/docker/build_pytorch_linux_build_image.sh"
    agents:
      "provider": "gcp"
      "machineType": "c2-standard-16"
    notify:
      - github_commit_status:
          context: "Build PyTorch Docker image"
  - wait
  - trigger: ml-cpp-pr-builds
    async: false
    build:
      branch: "${BUILDKITE_BRANCH}"
      commit: "${BUILDKITE_COMMIT}"
      message: "${BUILDKITE_MESSAGE}"
      env:
        DOCKER_IMAGE: "docker.elastic.co/ml-dev/ml-linux-dependency-build:pytorch_latest"
        GITHUB_PR_COMMENT_VAR_PLATFORM: "linux"
        GITHUB_PR_COMMENT_VAR_ARCH: "x86_64"
        GITHUB_PR_COMMENT_VAR_ACTION: "run_pytorch_tests"
        GITHUB_PR_TRIGGER_COMMENT: ""
EOL
