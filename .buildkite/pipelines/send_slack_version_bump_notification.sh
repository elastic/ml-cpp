#!/usr/bin/env bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
# Single Slack notification for the ml-cpp-version-bump pipeline: runs after the
# bump step opens the PR. Reads ml_cpp_version_bump_pr_url from Buildkite meta-data
# (set by dev-tools/bump_version.sh) and posts the PR link so reviewers can approve.
#
# Slack notify must live on the step (see Buildkite docs): build-level notify fires only
# on build.finished — after every downstream step including long DRA waits — so the
# message would appear hours late or never if someone checks earlier.
#
# Optional env:
#   ML_CPP_VERSION_BUMP_SLACK_CHANNEL — override channel (default #machine-learn-build)

set -euo pipefail

CHANNEL="${ML_CPP_VERSION_BUMP_SLACK_CHANNEL:-#machine-learn-build}"

if [[ "${BUILDKITE:-}" != "true" ]]; then
    echo "BUILDKITE is not true — skipping Slack notification (local run)."
    exit 0
fi

if ! command -v buildkite-agent >/dev/null 2>&1; then
    echo "ERROR: buildkite-agent not in PATH; cannot read meta-data or upload Slack notify pipeline." >&2
    echo "Use the same agent image as bump-version (Wolfi), not a minimal python image." >&2
    exit 1
fi

pr_url=""
changed="false"
pr_url=$(buildkite-agent meta-data get "ml_cpp_version_bump_pr_url" 2>/dev/null || true)
changed=$(buildkite-agent meta-data get "ml_cpp_version_bump_changed" 2>/dev/null || echo "false")
# Meta-data values must not contain stray whitespace (Breaks truthiness.)
pr_url=$(echo -n "${pr_url}" | tr -d '\r')
changed=$(echo -n "${changed}" | tr -d '\r')

if [[ -z "${pr_url}" && "${changed}" != "true" ]]; then
    echo "No version-bump PR opened (pr_url empty, ml_cpp_version_bump_changed=${changed}); skipping Slack notification."
    exit 0
fi

if [[ -z "${pr_url}" && "${changed}" == "true" ]]; then
    body_line="DRY RUN — no pull request URL (simulated bump)."
else
    body_line="Pull request (approval required): ${pr_url}"
fi

(
    cat <<EOF
steps:
  - label: "Schedule :slack: notification (version bump)"
    command: "echo schedule :slack: notification"
    notify:
      - slack:
          channels:
            - "${CHANNEL}"
          message: |
            **Version bump PR — approval required**
            ${body_line}
            Branch: \${BUILDKITE_BRANCH}
            NEW_VERSION: \${NEW_VERSION:-"(unset)"}
            BRANCH (param): \${BRANCH:-"(unset)"}
            VERSION_BUMP_MERGE_AUTO: \${VERSION_BUMP_MERGE_AUTO:-"(unset)"}
            DRY_RUN: \${DRY_RUN:-"(unset)"}
            Pipeline: \${BUILDKITE_BUILD_URL}
            Build: \${BUILDKITE_BUILD_NUMBER}
            Please review and approve this pull request so it can merge (subject to branch protection).
EOF
) | buildkite-agent pipeline upload
