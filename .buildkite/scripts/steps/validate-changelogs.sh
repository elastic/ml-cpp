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

SKIP_LABELS=">test >refactoring >docs >build >non-issue"

# On PR builds, check if the PR has a label that skips changelog validation.
# BUILDKITE_PULL_REQUEST_LABELS is a comma-separated list set by Buildkite.
if [[ -n "${BUILDKITE_PULL_REQUEST_LABELS:-}" ]]; then
  IFS=',' read -ra LABELS <<< "${BUILDKITE_PULL_REQUEST_LABELS}"
  for label in "${LABELS[@]}"; do
    label="$(echo "${label}" | xargs)"  # trim whitespace
    for skip in ${SKIP_LABELS}; do
      if [[ "${label}" == "${skip}" ]]; then
        echo "Skipping changelog validation: PR has label '${label}'"
        exit 0
      fi
    done
  done
fi

# Install Python dependencies
pip3 install --quiet pyyaml jsonschema 2>/dev/null || pip install --quiet pyyaml jsonschema

# Find changelog files changed in this PR (compared to main/target branch)
TARGET_BRANCH="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"

# Fetch the target branch so we can diff against it
git fetch origin "${TARGET_BRANCH}" --depth=1 2>/dev/null || true

CHANGED_CHANGELOGS=$(git diff --name-only --diff-filter=ACM "origin/${TARGET_BRANCH}"...HEAD -- 'docs/changelog/*.yaml' || true)

if [[ -z "${CHANGED_CHANGELOGS}" ]]; then
  echo "No changelog files found in this PR."
  echo "If this PR changes user-visible behaviour, please add a changelog entry."
  echo "See docs/changelog/README.md for details."
  echo "To skip this check, add one of these labels: ${SKIP_LABELS}"

  # Soft warning rather than hard failure during rollout
  if [[ "${CHANGELOG_REQUIRED:-false}" == "true" ]]; then
    exit 1
  fi
  exit 0
fi

echo "Validating changelog files:"
echo "${CHANGED_CHANGELOGS}"
echo ""

python3 dev-tools/validate_changelogs.py ${CHANGED_CHANGELOGS}
