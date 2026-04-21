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

# Shared logic to choose which elasticsearch fork/branch to use for ml-cpp CI:
# integration test clones (run_es_tests_common.sh) and Buildkite pipelines that
# need ELASTICSEARCH_SUBMODULE_COMMIT without cloning.
#
# Source this file, then call pickCloneTarget. It reads (in order of precedence):
#   GITHUB_PR_OWNER / GITHUB_PR_BRANCH — when the job is tied to a GitHub PR
#   PR_AUTHOR / PR_SOURCE_BRANCH — fork and branch for coordinated ml-cpp + ES changes
#   elastic / PR_SOURCE_BRANCH — upstream branch matching the ml-cpp PR branch name
#   elastic / PR_TARGET_BRANCH — target branch of the ml-cpp PR
#   elastic / main — final fallback
#
# On success, SELECTED_FORK and SELECTED_BRANCH are set. Optional helper
# elasticsearch_selected_branch_head_sha prints the remote HEAD commit for that
# pair (same transport as isCloneTargetValid: git@github.com).
#
# This file must be sourced (not executed) so that SELECTED_* remain in the caller's shell.

function isCloneTargetValid {
    FORK_TO_CHECK="$1"
    BRANCH_TO_CHECK="$2"
    # Diagnostics must go to stderr: callers (e.g. deploy_serverless_qa.yml.sh)
    # pipe stdout to `buildkite-agent pipeline upload` and expect only YAML.
    echo "Checking for '$BRANCH_TO_CHECK' branch at $FORK_TO_CHECK/elasticsearch" >&2
    if [ -n "$(git ls-remote --heads "git@github.com:$FORK_TO_CHECK/elasticsearch.git" "$BRANCH_TO_CHECK" 2>/dev/null)" ]; then
        echo "Will use '$BRANCH_TO_CHECK' branch at $FORK_TO_CHECK/elasticsearch for ES integration tests" >&2
        return 0
    fi
    return 1
}

SELECTED_FORK=elastic
SELECTED_BRANCH=main

function pickCloneTarget {

    if isCloneTargetValid "$GITHUB_PR_OWNER" "$GITHUB_PR_BRANCH" ; then
        SELECTED_FORK="$GITHUB_PR_OWNER"
        SELECTED_BRANCH="$GITHUB_PR_BRANCH"
        return 0
    fi

    if isCloneTargetValid "$PR_AUTHOR" "$PR_SOURCE_BRANCH" ; then
        SELECTED_FORK="$PR_AUTHOR"
        SELECTED_BRANCH="$PR_SOURCE_BRANCH"
        return 0
    fi

    if isCloneTargetValid "$SELECTED_FORK" "$PR_SOURCE_BRANCH" ; then
        SELECTED_BRANCH="$PR_SOURCE_BRANCH"
        return 0
    fi

    if isCloneTargetValid "$SELECTED_FORK" "$PR_TARGET_BRANCH" ; then
        SELECTED_BRANCH="$PR_TARGET_BRANCH"
        return 0
    fi

    if isCloneTargetValid "$SELECTED_FORK" "$SELECTED_BRANCH" ; then
        return 0
    fi

    return 1
}

# Prints the commit SHA at the head of SELECTED_BRANCH on SELECTED_FORK, or empty if unavailable.
function elasticsearch_selected_branch_head_sha {
    git ls-remote --heads "git@github.com:${SELECTED_FORK}/elasticsearch.git" "${SELECTED_BRANCH}" 2>/dev/null | awk '{print $1; exit}'
}
