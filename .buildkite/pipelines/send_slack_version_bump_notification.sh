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
# Set ML_CPP_VERSION_BUMP_TEST_MODE to any non-empty value to prepend a loud
# "TEST RUN" banner and optional custom channel (see below).
#
# Optional env:
#   ML_CPP_VERSION_BUMP_TEST_MODE   — non-empty => test banner + wording
#   ML_CPP_VERSION_BUMP_SLACK_CHANNEL — override channel (default #machine-learn-build)

CHANNEL="${ML_CPP_VERSION_BUMP_SLACK_CHANNEL:-#machine-learn-build}"

if [ -n "${ML_CPP_VERSION_BUMP_TEST_MODE:-}" ]; then
    TEST_LINES='        :rotating_light: **TEST RUN — ml-cpp version bump pipeline** :rotating_light:
        _This is not a production release._ (ML_CPP_VERSION_BUMP_TEST_MODE is set on the build.)
        Set ML_CPP_VERSION_BUMP_SKIP_DRA_WAIT on the build to skip artifact polling for short smoke tests.

'
else
    TEST_LINES=""
fi

cat <<EOL
steps:
  - label: "Schedule :slack: notification (version bump)"
    command: "echo schedule :slack: notification"
notify:
  - slack:
      channels:
        - "${CHANNEL}"
      message: |
${TEST_LINES}        **Version bump pipeline**
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
