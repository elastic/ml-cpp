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
#   `gh auth login` (preferred for local runs), VAULT_GITHUB_TOKEN (CI), or GH_TOKEN.
#   GITHUB_TOKEN is intentionally ignored so a stale shell export does not override
#   an interactive gh login session.
# If gh is missing, dev-tools/ensure_github_cli.sh runs (Wolfi apk, else Linux
# tarball) unless SKIP_GH_AUTO_INSTALL=true.
#
# Usage:
#   create_github_pull_request.sh --repo ORG/REPO --base BASE --head HEAD \
#       --title T --body B [--label NAME] [--merge | --merge-auto] [--merge-method merge|squash|rebase]
#
# On success, prints the PR URL to stdout (single line). Merge progress to stderr.
#
#   --merge       Squash/merge/rebase immediately after create (subject to branch rules).
#   --merge-auto  Enable GitHub auto-merge with the chosen merge method (same pattern as
#                 backport workflow: gh pr merge --auto --squash). Merge occurs when
#                 required checks pass; no human click needed if policies allow the bot.
#
# Environment:
#   VERSION_BUMP_MERGE_METHOD — merge style for --merge / --merge-auto (merge|squash|rebase).
#     Default squash — elastic/ml-cpp forbids merge commits on protected branches.
#   VERSION_BUMP_MERGE_ADMIN — set to true to add gh pr merge --admin (bypasses some rule
#     violations only if the token identity has appropriate admin/bypass rights on the repo).
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

REPO=""
BASE=""
HEAD_REF=""
TITLE=""
BODY=""
DO_MERGE="false"
DO_MERGE_AUTO="false"
MERGE_METHOD="${VERSION_BUMP_MERGE_METHOD:-squash}"
LABELS=()

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
        --label)
            LABELS+=("$2")
            shift 2
            ;;
        --merge)
            DO_MERGE="true"
            shift 1
            ;;
        --merge-auto)
            DO_MERGE_AUTO="true"
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

if [[ "$DO_MERGE" == "true" && "$DO_MERGE_AUTO" == "true" ]]; then
    echo "ERROR: use only one of --merge or --merge-auto." >&2
    exit 1
fi

if [[ -z "$REPO" || -z "$BASE" || -z "$HEAD_REF" || -z "$TITLE" || -z "$BODY" ]]; then
    echo "ERROR: --repo, --base, --head, --title, and --body are required." >&2
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

# gh prefers GH_TOKEN over `gh auth login` credentials. Unset GH_TOKEN when the CLI
# is already logged in so local testing works with `gh auth login` even if GITHUB_TOKEN
# is exported in the shell. GITHUB_TOKEN is never used as a fallback.
if gh auth status >/dev/null 2>&1; then
    unset GH_TOKEN
elif [[ -z "${GH_TOKEN:-}" && -n "${VAULT_GITHUB_TOKEN:-}" ]]; then
    export GH_TOKEN="${VAULT_GITHUB_TOKEN}"
fi

if ! gh auth status >/dev/null 2>&1; then
    echo "ERROR: gh is not authenticated. Run \`gh auth login\` or set VAULT_GITHUB_TOKEN / GH_TOKEN." >&2
    exit 1
fi

declare -a create_cmd=(
    gh pr create
    --repo "$REPO"
    --base "$BASE"
    --head "$HEAD_REF"
    --title "$TITLE"
    --body "$BODY"
)
if ((${#LABELS[@]} > 0)); then
    for label in "${LABELS[@]}"; do
        create_cmd+=(--label "$label")
    done
fi

PR_URL=$("${create_cmd[@]}")

echo "$PR_URL"

if [[ "$DO_MERGE" == "true" || "$DO_MERGE_AUTO" == "true" ]]; then
    # Older packaged gh (e.g. Wolfi apk) does not support --yes on pr merge; rely on
    # non-TTY / GH_PROMPT_DISABLED for unattended merges.
    declare -a merge_admin=()
    if [[ "${VERSION_BUMP_MERGE_ADMIN:-}" == "true" ]]; then
        merge_admin+=(--admin)
    fi
    if [[ "$DO_MERGE_AUTO" == "true" ]]; then
        GH_PROMPT_DISABLED=1 gh pr merge "$PR_URL" --auto "${MERGE_TYPE[@]}" "${merge_admin[@]}"
        echo "Enabled auto-merge: ${PR_URL}" >&2
    else
        GH_PROMPT_DISABLED=1 gh pr merge "$PR_URL" "${MERGE_TYPE[@]}" "${merge_admin[@]}"
        echo "Merged: ${PR_URL}" >&2
    fi
fi
