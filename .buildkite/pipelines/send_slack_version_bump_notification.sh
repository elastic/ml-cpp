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
#   ML_CPP_VERSION_BUMP_SLACK_CHANNEL — override channel(s). Comma-separated list
#     for multiple channels (default "#machine-learn-build,#ml-core").

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../dev-tools/version_bump_lib.sh
source "${REPO_ROOT}/dev-tools/version_bump_lib.sh"

# Buildkite's slack notify accepts a list of channels, so a single build can fan
# the notification out to several. Accept a comma-separated override and render
# each entry as its own YAML list item (CHANNELS_YAML) injected into the notify
# block below. The Buildkite Slack app must be connected to every channel named
# here or the post is silently dropped for that channel.
CHANNELS_RAW="${ML_CPP_VERSION_BUMP_SLACK_CHANNEL:-#machine-learn-build,#ml-core}"
CHANNELS_YAML=""
# Each entry is interpolated into the generated pipeline YAML inside double
# quotes, so restrict it to characters valid for a Slack notify target: #channel,
# @user, a channel/user ID, or Buildkite's [token]#channel form. Rejecting
# anything else (quotes, backslashes, newlines, spaces, ...) fails fast on a
# malformed override rather than emitting invalid YAML or allowing YAML/step
# injection. ']' is first and '-' last in the class so both are literal, and no
# backslash appears in the class so '\' is not accepted.
_channel_allowed='^[]A-Za-z0-9_.#@[-]+$'
IFS=',' read -ra _channels <<<"${CHANNELS_RAW}"
for _ch in "${_channels[@]}"; do
    _ch="$(version_bump_trim_value "${_ch}")"
    [[ -z "${_ch}" ]] && continue
    if [[ ! "${_ch}" =~ $_channel_allowed ]]; then
        echo "ERROR: invalid Slack channel '${_ch}' in ML_CPP_VERSION_BUMP_SLACK_CHANNEL; allowed characters: letters, digits and '# @ _ . - [ ]'." >&2
        exit 1
    fi
    CHANNELS_YAML+="            - \"${_ch}\""$'\n'
done
CHANNELS_YAML="${CHANNELS_YAML%$'\n'}"
if [[ -z "${CHANNELS_YAML}" ]]; then
    echo "ERROR: no Slack channels resolved from ML_CPP_VERSION_BUMP_SLACK_CHANNEL='${CHANNELS_RAW}'." >&2
    exit 1
fi

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
minor_branch_created="false"
workflow="$(version_bump_trim_value "${WORKFLOW:-patch}")"
pr_url=$(buildkite-agent meta-data get "ml_cpp_version_bump_pr_url" 2>/dev/null || true)
changed=$(buildkite-agent meta-data get "ml_cpp_version_bump_changed" 2>/dev/null || echo "false")
minor_branch_created=$(buildkite-agent meta-data get "ml_cpp_minor_branch_created" 2>/dev/null || echo "false")
# Meta-data values must not contain stray whitespace (breaks truthiness.)
pr_url="$(version_bump_trim_value "${pr_url}")"
changed="$(version_bump_trim_value "${changed}")"
minor_branch_created="$(version_bump_trim_value "${minor_branch_created}")"

if [[ "${workflow}" == "minor" ]]; then
    if [[ "${minor_branch_created}" != "true" && "${changed}" != "true" ]]; then
        echo "Minor freeze: no branch created and no main-bump PR; skipping Slack notification."
        exit 0
    fi
    branch_line="Release branch \${BRANCH:-\"(unset)\"} created (or already present) at \${NEW_VERSION:-\"(unset)\"}."
    if [[ -n "${pr_url}" ]]; then
        pr_line="Main bump pull request (approval required): ${pr_url}"
    elif [[ "${changed}" == "true" ]]; then
        pr_line="DRY RUN — main bump PR simulated (no URL)."
    else
        pr_line="Main bump — no PR required (already at derived next minor)."
    fi
    slack_title="**Minor feature freeze — action may be required**"
    (
        cat <<EOF
steps:
  - label: "Schedule :slack: notification (version bump)"
    command: "echo schedule :slack: notification"
    notify:
      - slack:
          channels:
${CHANNELS_YAML}
          message: |
            <!subteam^S76JPTCBE|ml-team>
            ${slack_title}
            ${branch_line}
            ${pr_line}
            WORKFLOW: \${WORKFLOW:-"(unset)"}
            Branch: \${BUILDKITE_BRANCH}
            NEW_VERSION: \${NEW_VERSION:-"(unset)"}
            BRANCH (param): \${BRANCH:-"(unset)"}
            VERSION_BUMP_MERGE_AUTO: \${VERSION_BUMP_MERGE_AUTO:-"(unset)"}
            DRY_RUN: \${DRY_RUN:-"(unset)"}
            Pipeline: \${BUILDKITE_BUILD_URL}
            Build: \${BUILDKITE_BUILD_NUMBER}
            Please review and approve the main bump pull request when present (subject to branch protection).
EOF
    ) | buildkite-agent pipeline upload
    exit 0
else
    if [[ -z "${pr_url}" && "${changed}" != "true" ]]; then
        echo "No version-bump PR opened (pr_url empty, ml_cpp_version_bump_changed=${changed}); skipping Slack notification."
        exit 0
    fi

    if [[ -z "${pr_url}" && "${changed}" == "true" ]]; then
        body_line="DRY RUN — no pull request URL (simulated bump)."
    else
        body_line="Pull request (approval required): ${pr_url}"
    fi
    slack_title="**Version bump PR — approval required**"
    slack_body="${body_line}"
fi

(
    cat <<EOF
steps:
  - label: "Schedule :slack: notification (version bump)"
    command: "echo schedule :slack: notification"
    notify:
      - slack:
          channels:
${CHANNELS_YAML}
          message: |
            <!subteam^S76JPTCBE|ml-team>
            ${slack_title}
            ${slack_body}
            WORKFLOW: \${WORKFLOW:-"(unset)"}
            Branch: \${BUILDKITE_BRANCH}
            NEW_VERSION: \${NEW_VERSION:-"(unset)"}
            BRANCH (param): \${BRANCH:-"(unset)"}
            VERSION_BUMP_MERGE_AUTO: \${VERSION_BUMP_MERGE_AUTO:-"(unset)"}
            DRY_RUN: \${DRY_RUN:-"(unset)"}
            Pipeline: \${BUILDKITE_BUILD_URL}
            Build: \${BUILDKITE_BUILD_NUMBER}
            Please review and approve the main bump pull request when present (subject to branch protection).
EOF
) | buildkite-agent pipeline upload
