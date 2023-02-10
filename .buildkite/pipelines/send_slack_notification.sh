#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
# Create a unique job that sends slack notification only.
cat <<EOL
steps:
  - label: "Schedule :slack: notification"
    command: "echo schedule :slack: notification"
notify:
  - slack:
      channels:
        - “#machine-learn-build"
      message: |
        Branch: ${BUILDKITE_BRANCH}
        User: ${BUILDKITE_BUILD_CREATOR}
        Pipeline: ${BUILDKITE_BUILD_URL}
        Build: ${BUILDKITE_BUILD_NUMBER}
    if: build.pull_request.id == null
EOL
