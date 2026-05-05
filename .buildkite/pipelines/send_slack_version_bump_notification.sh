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
# Slack notifications for the ml-cpp-version-bump pipeline only (not PR builds).
#
# Optional env:
#   ML_CPP_VERSION_BUMP_SLACK_CHANNEL — override channel (default #machine-learn-build)

CHANNEL="${ML_CPP_VERSION_BUMP_SLACK_CHANNEL:-#machine-learn-build}"

cat <<EOL
steps:
  - label: "Schedule :slack: notification (version bump)"
    command: "echo schedule :slack: notification"
notify:
  - slack:
      channels:
        - "${CHANNEL}"
      message: |
        **Version bump pipeline**
        Build message: \${BUILDKITE_MESSAGE:-"(none)"}
        Branch: \${BUILDKITE_BRANCH}
        User: \${BUILDKITE_BUILD_CREATOR}
        NEW_VERSION: \${NEW_VERSION:-"(unset)"}
        BRANCH (param): \${BRANCH:-"(unset)"}
        DRY_RUN: \${DRY_RUN:-"(unset)"}
        Pipeline: \${BUILDKITE_BUILD_URL}
        Build: \${BUILDKITE_BUILD_NUMBER}
    if: build.pull_request.id == null
EOL
