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
  - label: "Validate formatting with clang-format"
    key: "check_style"
    command: ".buildkite/scripts/steps/check-style.sh --all"
    agents:
      image: "docker.elastic.co/ml-dev/ml-check-style:2"
    notify:
      - github_commit_status:
          context: "Validate formatting with clang-format"
  - label: "Validate changelog entries"
    key: "validate_changelogs"
    command: ".buildkite/scripts/steps/validate-changelogs.sh"
    agents:
      # bookworm (not slim): Buildkite agent environment hooks need curl + git before the step runs
      image: "python:3.11-bookworm"
    soft_fail: true
    notify:
      - github_commit_status:
          context: "Validate changelog entries"
  - label: "Unit tests: changelog Python tools"
    key: "test_changelog_tools"
    command: ".buildkite/scripts/steps/test-changelog-tools.sh"
    agents:
      image: "python:3.11-bookworm"
    notify:
      - github_commit_status:
          context: "Unit tests: changelog Python tools"
EOL
