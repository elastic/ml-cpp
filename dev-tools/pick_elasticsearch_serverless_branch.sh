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

# Choose which branch of elastic/elasticsearch-serverless to pass to Buildkite
# trigger steps (es-pr-check, deploy-qa). This mirrors the fork/branch *idea*
# behind pick_elasticsearch_clone_target.sh, but for the serverless repo.
#
# Source this file after setting (from ml-cpp PR metadata):
#   PR_AUTHOR_FORK  — fork owner from BUILDKITE_BRANCH (author before ':')
#   PR_SOURCE       — branch name from BUILDKITE_BRANCH (after ':')
#   PR_TARGET       — BUILDKITE_PULL_REQUEST_BASE_BRANCH (default main)
# Optional override:
#   ES_SERVERLESS_BRANCH — force this branch name
#
# Resolution runs in sequential steps (no nested if/elif chain) so a fork-only
# branch warning never blocks later fallbacks to elastic/ (PR_SOURCE, then
# PR_TARGET on elastic).
#
# Call pickElasticsearchServerlessBranch. It sets SERVERLESS_BRANCH and writes
# diagnostics to stderr (callers often pipe stdout to buildkite-agent).

SERVERLESS_BRANCH="main"

function isElasticsearchServerlessBranchAtRemote {
    local repo="$1"
    local branch="$2"
    [ -n "$branch" ] && git ls-remote --heads "https://github.com/${repo}/elasticsearch-serverless.git" "$branch" 2>/dev/null | grep -q .
}

function pickElasticsearchServerlessBranch {
    SERVERLESS_BRANCH="main"

    # 1) Explicit override (must exist on elastic/)
    if [ -n "${ES_SERVERLESS_BRANCH:-}" ]; then
        if isElasticsearchServerlessBranchAtRemote "elastic" "${ES_SERVERLESS_BRANCH}"; then
            SERVERLESS_BRANCH="${ES_SERVERLESS_BRANCH}"
            echo "Using explicit ES_SERVERLESS_BRANCH override: $SERVERLESS_BRANCH" >&2
            echo "Resolved elasticsearch-serverless branch: $SERVERLESS_BRANCH" >&2
            return 0
        fi
        echo "ERROR: ES_SERVERLESS_BRANCH override '${ES_SERVERLESS_BRANCH}' was not found on elastic/elasticsearch-serverless." >&2
        echo "Set ES_SERVERLESS_BRANCH to an existing branch on elastic/elasticsearch-serverless." >&2
        return 1
    fi

    # 2) Prefer PR_SOURCE when it exists on elastic/ (Buildkite only consumes elastic/)
    if isElasticsearchServerlessBranchAtRemote "elastic" "$PR_SOURCE"; then
        SERVERLESS_BRANCH="$PR_SOURCE"
        if [ -n "$PR_AUTHOR_FORK" ] && isElasticsearchServerlessBranchAtRemote "$PR_AUTHOR_FORK" "$PR_SOURCE"; then
            echo "Found '$PR_SOURCE' on both $PR_AUTHOR_FORK and elastic; using elastic/" >&2
        fi
    elif [ -n "$PR_AUTHOR_FORK" ] && isElasticsearchServerlessBranchAtRemote "$PR_AUTHOR_FORK" "$PR_SOURCE"; then
        echo "WARNING: Found '$PR_SOURCE' on $PR_AUTHOR_FORK/elasticsearch-serverless but not on elastic/." >&2
        echo "The trigger step can only use branches on elastic/elasticsearch-serverless." >&2
        echo "Push the branch to elastic/ or set ES_SERVERLESS_BRANCH explicitly." >&2
    fi

    # 3) Still unresolved: fall back to PR base branch on elastic/ when available
    if [ "$SERVERLESS_BRANCH" = "main" ] && [ -n "${PR_TARGET:-}" ] && [ "$PR_TARGET" != "main" ]; then
        if isElasticsearchServerlessBranchAtRemote "elastic" "$PR_TARGET"; then
            SERVERLESS_BRANCH="$PR_TARGET"
            echo "Using elasticsearch-serverless branch '$PR_TARGET' from PR base (elastic/) as fallback." >&2
        fi
    fi

    echo "Resolved elasticsearch-serverless branch: $SERVERLESS_BRANCH" >&2
}
