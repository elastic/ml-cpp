#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

set -euo pipefail

SKIP_LABELS="|>test|>refactoring|>docs|>build|>non-issue|"

# On PR builds, check if the PR has a label that skips changelog validation.
# BUILDKITE_PULL_REQUEST_LABELS is a comma-separated list set by Buildkite.
if [[ -n "${BUILDKITE_PULL_REQUEST_LABELS:-}" ]]; then
  IFS=',' read -ra LABELS <<< "${BUILDKITE_PULL_REQUEST_LABELS}"
  for label in "${LABELS[@]}"; do
    label="$(echo "${label}" | xargs)"  # trim whitespace
    if [[ "${SKIP_LABELS}" == *"|${label}|"* ]]; then
      echo "Skipping changelog validation: PR has label '${label}'"
      exit 0
    fi
  done
fi

# Install system and Python dependencies
if ! command -v git &>/dev/null; then
  apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1
fi
python3 -m pip install --quiet --break-system-packages pyyaml jsonschema 2>/dev/null \
  || python3 -m pip install --quiet pyyaml jsonschema

# Find changelog files changed in this PR (compared to main/target branch)
TARGET_BRANCH="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"

# Fetch the target branch so we can diff against it
if ! git fetch origin "${TARGET_BRANCH}" --depth=1 2>/dev/null; then
  echo "Warning: could not fetch origin/${TARGET_BRANCH}, skipping changelog validation"
  exit 0
fi

if ! git rev-parse --verify "origin/${TARGET_BRANCH}" >/dev/null 2>&1; then
  echo "Warning: origin/${TARGET_BRANCH} not available, skipping changelog validation"
  exit 0
fi

CHANGED_CHANGELOGS=$(git diff --name-only --diff-filter=ACM "origin/${TARGET_BRANCH}"...HEAD -- 'docs/changelog/*.yaml')
DIFF_EXIT=$?
if [[ $DIFF_EXIT -ne 0 ]]; then
  echo "Warning: git diff failed (exit $DIFF_EXIT), skipping changelog validation"
  exit 0
fi

if [[ -z "${CHANGED_CHANGELOGS}" ]]; then
  echo "No changelog files found in this PR."
  echo "If this PR changes user-visible behaviour, please add a changelog entry."
  echo "See docs/changelog/README.md for details."
  echo "To skip this check, add one of these labels: >test, >refactoring, >docs, >build, >non-issue"

  # Soft warning rather than hard failure during rollout
  if [[ "${CHANGELOG_REQUIRED:-false}" == "true" ]]; then
    exit 1
  fi
  exit 0
fi

echo "Validating changelog files:"
echo "${CHANGED_CHANGELOGS}"
echo ""

readarray -t CHANGED_FILES <<< "${CHANGED_CHANGELOGS}"
python3 dev-tools/validate_changelogs.py "${CHANGED_FILES[@]}"
