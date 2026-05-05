#!/bin/bash
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
# Create a pull request (and optionally merge) using the GitHub CLI.
#
# Requires: gh (https://cli.github.com/) in PATH, authenticated via:
#   GITHUB_TOKEN, VAULT_GITHUB_TOKEN, or GH_TOKEN
# If gh is missing, dev-tools/ensure_github_cli.sh runs (Wolfi apk, else Linux
# tarball) unless SKIP_GH_AUTO_INSTALL=true.
#
# Usage:
#   create_github_pull_request.sh --repo ORG/REPO --base BASE --head HEAD \
#       --title T --body B [--merge] [--merge-method merge|squash|rebase]
#
# On success, prints the PR URL to stdout (single line). Merge progress to stderr.
#
# Environment:
#   VERSION_BUMP_MERGE_METHOD — merge style when --merge is used (merge|squash|rebase).
#     Default squash — elastic/ml-cpp forbids merge commits on protected branches.
#   SKIP_GH_AUTO_INSTALL, GH_CLI_VERSION, GH_CLI_INSTALL_PREFIX — see ensure_github_cli.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v gh >/dev/null 2>&1; then
    "${SCRIPT_DIR}/ensure_github_cli.sh"
fi

if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: GitHub CLI (gh) is not available; see dev-tools/ensure_github_cli.sh" >&2
    exit 1
fi

# gh honors GH_TOKEN; map Vault/standard env the same as other automation
if [[ -z "${GH_TOKEN:-}" ]]; then
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        export GH_TOKEN="${GITHUB_TOKEN}"
    elif [[ -n "${VAULT_GITHUB_TOKEN:-}" ]]; then
        export GH_TOKEN="${VAULT_GITHUB_TOKEN}"
    fi
fi

if [[ -z "${GH_TOKEN:-}" ]]; then
    echo "ERROR: Set GITHUB_TOKEN, VAULT_GITHUB_TOKEN, or GH_TOKEN for gh auth." >&2
    exit 1
fi

REPO=""
BASE=""
HEAD_REF=""
TITLE=""
BODY=""
DO_MERGE="false"
MERGE_METHOD="${VERSION_BUMP_MERGE_METHOD:-squash}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)
            REPO="$2"
            shift 2
            ;;
        --base)
            BASE="$2"
            shift 2
            ;;
        --head)
            HEAD_REF="$2"
            shift 2
            ;;
        --title)
            TITLE="$2"
            shift 2
            ;;
        --body)
            BODY="$2"
            shift 2
            ;;
        --merge)
            DO_MERGE="true"
            shift 1
            ;;
        --merge-method)
            MERGE_METHOD="$2"
            shift 2
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$REPO" || -z "$BASE" || -z "$HEAD_REF" || -z "$TITLE" ]]; then
    echo "ERROR: --repo, --base, --head, and --title are required." >&2
    exit 1
fi

case "$MERGE_METHOD" in
    merge) MERGE_TYPE=(--merge) ;;
    squash) MERGE_TYPE=(--squash) ;;
    rebase) MERGE_TYPE=(--rebase) ;;
    *)
        echo "ERROR: invalid merge method: ${MERGE_METHOD}" >&2
        exit 1
        ;;
esac

PR_URL=$(gh pr create \
    --repo "$REPO" \
    --base "$BASE" \
    --head "$HEAD_REF" \
    --title "$TITLE" \
    --body "$BODY")

echo "$PR_URL"

if [[ "$DO_MERGE" == "true" ]]; then
    # Older packaged gh (e.g. Wolfi apk) does not support --yes on pr merge; rely on
    # non-TTY / GH_PROMPT_DISABLED for unattended merges.
    GH_PROMPT_DISABLED=1 gh pr merge "$PR_URL" "${MERGE_TYPE[@]}"
    echo "Merged: ${PR_URL}" >&2
fi
