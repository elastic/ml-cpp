#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

. .buildkite/scripts/common/base.sh

cat <<EOL
steps:
  - label: "Upload artifacts to S3 :s3:"
    key: "upload_artifacts"
    depends_on: "java_integration_tests"
    command:
      - 'buildkite-agent artifact download "build/*" .'
      - 'echo "${RED}This step is disabled until BuildKite migration is complete${NOCOLOR}"'
      #- "./.buildkite/scripts/steps/upload_to_s3.sh"
    agents:
      cpu: "2"
      ephemeralStorage: "20G"
      memory: "4G"
      image: "docker.elastic.co/ml-dev/ml_cpp_linux_x86_64_jdk17:2"
      # Run as a non-root user
      imageUID: "1000"
EOL
