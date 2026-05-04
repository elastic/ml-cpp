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
# Validates NEW_VERSION / BRANCH / WORKFLOW against elasticsearchVersion on the
# remote release branch before ml-cpp-version-bump runs bump_version.sh.
# Semantic rules live in version_bump_validation.py (unit-tested).
#
# Environment:
#   NEW_VERSION — required target stack version (MAJOR.MINOR.PATCH), unless skipped
#   BRANCH — required release branch (e.g. 9.5), unless skipped
#   WORKFLOW — patch (default) or minor
#   SKIP_VERSION_VALIDATION — set to "true" to skip (emergency override only)
#   PYTHON — interpreter (default: python3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VALIDATION_PY="${SCRIPT_DIR}/version_bump_validation.py"

SKIP_VERSION_VALIDATION="${SKIP_VERSION_VALIDATION:-false}"

if [[ "$SKIP_VERSION_VALIDATION" == "true" ]]; then
    echo "WARNING: SKIP_VERSION_VALIDATION=true — version increment checks skipped." >&2
    exit 0
fi

: "${NEW_VERSION:?NEW_VERSION must be set}"
: "${BRANCH:?BRANCH must be set}"

WORKFLOW="${WORKFLOW:-patch}"

echo "=== Version bump validation ==="
echo "WORKFLOW:     ${WORKFLOW}"
echo "NEW_VERSION:  ${NEW_VERSION}"
echo "BRANCH:       ${BRANCH}"

echo "Fetching origin/${BRANCH}..."
git fetch origin "$BRANCH"

if ! git cat-file -e FETCH_HEAD:gradle.properties 2>/dev/null; then
    echo "ERROR: gradle.properties missing at FETCH_HEAD (origin/${BRANCH})" >&2
    exit 1
fi

CURRENT_VERSION=$(
    git show FETCH_HEAD:gradle.properties | grep '^elasticsearchVersion=' | head -1 | cut -d= -f2 | tr -d '[:space:]'
)

if [[ -z "$CURRENT_VERSION" ]]; then
    echo "ERROR: could not read elasticsearchVersion from origin/${BRANCH} gradle.properties" >&2
    exit 1
fi

echo "Current version on origin/${BRANCH}: ${CURRENT_VERSION}"

exec "$PYTHON" "$VALIDATION_PY" validate-and-report \
    --current "$CURRENT_VERSION" \
    --new "$NEW_VERSION" \
    --branch "$BRANCH" \
    --workflow "$WORKFLOW"
