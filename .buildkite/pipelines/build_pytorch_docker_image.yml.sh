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
  - label: "Build PyTorch Docker image"
    key: "build_pytorch_docker_image"
    command: "./dev-tools/docker/build_pytorch_linux_build_image.sh"
    agents:
      "cpu": "6",
      "ephemeralStorage": "20G",
      "memory": "64G",
      "image": "docker.elastic.co/ml-dev/ml-linux-build:29"
    notify:
      - github_commit_status:
          context: "Build PyTorch Docker image"
EOL
