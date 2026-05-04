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
# Local helper to exercise dev-tools/validate_version_bump_params.sh against
# origin/<BRANCH>: fetch the branch, derive NEW_VERSION from the current
# elasticsearchVersion in gradle.properties, export env, run the validator.
#
# Usage:
#   ./dev-tools/test_validate_version_bump_local.sh <BRANCH> [options]
#
# Options:
#   --negative       Use an intentionally invalid NEW_VERSION (expects validator failure)
#   --workflow patch | minor   Override workflow (default: patch for positive,
#                    minor mode targets MAJOR.(MINOR+1).0 on the next minor line)
#   --dry-run        Print env and computed versions only; do not run validator
#
# Examples:
#   ./dev-tools/test_validate_version_bump_local.sh 9.5
#   ./dev-tools/test_validate_version_bump_local.sh 9.5 --negative
#   ./dev-tools/test_validate_version_bump_local.sh 9.5 --workflow minor --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VALIDATOR="${SCRIPT_DIR}/validate_version_bump_params.sh"

usage() {
    echo "Usage: $0 <BRANCH> [--negative] [--workflow patch|minor] [--dry-run]" >&2
    echo "  BRANCH  Release branch MAJOR.MINOR (e.g. 9.5), matching origin." >&2
    exit 1
}

BRANCH=""
NEGATIVE="false"
WORKFLOW_OVERRIDE=""
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --negative)
            NEGATIVE="true"
            shift
            ;;
        --workflow)
            WORKFLOW_OVERRIDE="${2:?}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h | --help)
            usage
            ;;
        *)
            if [[ -n "$BRANCH" ]]; then
                echo "ERROR: unexpected argument: $1" >&2
                usage
            fi
            BRANCH="$1"
            shift
            ;;
    esac
done

if [[ -z "$BRANCH" ]]; then
    usage
fi

cd "$REPO_ROOT"

if [[ ! -x "$VALIDATOR" && ! -f "$VALIDATOR" ]]; then
    echo "ERROR: validator not found at ${VALIDATOR}" >&2
    exit 1
fi

parse_triple() {
    local v="$1"
    local re='^([0-9]+)\.([0-9]+)\.([0-9]+)$'
    if [[ "$v" =~ $re ]]; then
        _M="${BASH_REMATCH[1]}"
        _N="${BASH_REMATCH[2]}"
        _P="${BASH_REMATCH[3]}"
        return 0
    fi
    return 1
}

echo "=== Fetch origin/${BRANCH} ==="
git fetch origin "$BRANCH"

CURRENT_VERSION=$(
    git show FETCH_HEAD:gradle.properties | grep '^elasticsearchVersion=' | head -1 | cut -d= -f2 | tr -d '[:space:]'
)

if [[ -z "$CURRENT_VERSION" ]]; then
    echo "ERROR: could not read elasticsearchVersion from FETCH_HEAD" >&2
    exit 1
fi

if ! parse_triple "$CURRENT_VERSION"; then
    echo "ERROR: invalid elasticsearchVersion on branch: '${CURRENT_VERSION}'" >&2
    exit 1
fi

MAJOR="$_M"
MINOR="$_N"
PATCH="$_P"

WORKFLOW="${WORKFLOW_OVERRIDE:-patch}"

if [[ "$WORKFLOW" != "patch" && "$WORKFLOW" != "minor" ]]; then
    echo "ERROR: --workflow must be patch or minor" >&2
    exit 1
fi

if [[ "$NEGATIVE" == "true" && "$WORKFLOW" == "minor" ]]; then
    echo "ERROR: --negative is only implemented for patch workflow" >&2
    exit 1
fi

if [[ "$WORKFLOW" == "patch" ]]; then
    if [[ "$BRANCH" != "${MAJOR}.${MINOR}" ]]; then
        echo "ERROR: BRANCH '${BRANCH}' must match MAJOR.MINOR of current version (${MAJOR}.${MINOR}) for patch test" >&2
        exit 1
    fi
    if [[ "$NEGATIVE" == "true" ]]; then
        NEW_VERSION="${MAJOR}.${MINOR}.$((PATCH + 2))"
        echo "=== Negative test: invalid patch jump ${CURRENT_VERSION} → ${NEW_VERSION} (expected failure) ==="
    else
        NEW_VERSION="${MAJOR}.${MINOR}.$((PATCH + 1))"
        echo "=== Positive patch test: ${CURRENT_VERSION} → ${NEW_VERSION} ==="
    fi
else
    EXPECT_BRANCH="${MAJOR}.$((MINOR + 1))"
    if [[ "$BRANCH" != "$EXPECT_BRANCH" ]]; then
        echo "ERROR: for minor workflow, BRANCH must be '${EXPECT_BRANCH}' (next minor line); tip has ${CURRENT_VERSION} on origin/${BRANCH}" >&2
        exit 1
    fi
    NEW_VERSION="${MAJOR}.$((MINOR + 1)).0"
    echo "=== Positive minor test: ${CURRENT_VERSION} → ${NEW_VERSION} (WORKFLOW=minor) ==="
fi

export BRANCH
export NEW_VERSION
export WORKFLOW
unset SKIP_VERSION_VALIDATION 2>/dev/null || true

echo "BRANCH=${BRANCH}"
echo "NEW_VERSION=${NEW_VERSION}"
echo "WORKFLOW=${WORKFLOW}"
echo "CURRENT (origin/${BRANCH})=${CURRENT_VERSION}"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "=== --dry-run: not invoking validator ==="
    exit 0
fi

echo "=== Running validate_version_bump_params.sh ==="
set +e
"$VALIDATOR"
RC=$?
set -e

if [[ "$NEGATIVE" == "true" ]]; then
    if [[ "$RC" -eq 0 ]]; then
        echo "ERROR: negative test expected validator to fail, but it exited 0" >&2
        exit 1
    fi
    echo "OK: negative test — validator failed as expected (exit ${RC})"
    exit 0
fi

if [[ "$RC" -ne 0 ]]; then
    echo "ERROR: validator exited ${RC}" >&2
    exit "$RC"
fi

echo "OK: validator succeeded"
