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
# Validates NEW_VERSION / BRANCH against elasticsearchVersion on the
# remote release branch before ml-cpp-version-bump runs bump_version.sh.
# Semantic rules live in version_bump_validation.py (unit-tested).
#
# Environment:
#   NEW_VERSION — required target stack version (MAJOR.MINOR.PATCH), unless skipped
#   BRANCH — required release branch (e.g. 9.5), unless skipped
#   WORKFLOW — optional; defaults to patch. Supported: patch, minor (feature freeze).
#   SKIP_VERSION_VALIDATION — set to "true" to skip (emergency override only)
#   PYTHON — interpreter (default: python3)
#
# Buildkite (BUILDKITE=true): sets meta-data ml_cpp_version_bump_noop to true when
# origin/BRANCH already has NEW_VERSION, so downstream Slack/bump steps are skipped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=version_bump_lib.sh
source "${SCRIPT_DIR}/version_bump_lib.sh"

PYTHON="${PYTHON:-python3}"
VALIDATION_PY="${SCRIPT_DIR}/version_bump_validation.py"

SKIP_VERSION_VALIDATION="${SKIP_VERSION_VALIDATION:-false}"

if [[ "$SKIP_VERSION_VALIDATION" == "true" ]]; then
    echo "WARNING: SKIP_VERSION_VALIDATION=true — version increment checks skipped." >&2
    version_bump_set_noop_meta false
    exit 0
fi

: "${NEW_VERSION:?NEW_VERSION must be set}"
: "${BRANCH:?BRANCH must be set}"

NEW_VERSION="$(version_bump_trim_value "${NEW_VERSION}")"
BRANCH="$(version_bump_trim_value "${BRANCH}")"

WORKFLOW="${WORKFLOW:-patch}"
WORKFLOW="$(version_bump_trim_value "${WORKFLOW}")"

if [[ "$WORKFLOW" == "minor" ]]; then
    echo "=== Version bump validation (minor feature freeze) ==="
    echo "WORKFLOW:     ${WORKFLOW}"
    echo "NEW_VERSION:  ${NEW_VERSION} (expected on release branch ${BRANCH})"
    echo "BRANCH:       ${BRANCH} (release branch to create)"
    if [[ "$BRANCH" == testing-* ]]; then
        echo "              (sandbox: version rules use identity ${BRANCH#testing-})"
    fi

    echo "Fetching origin/main and checking origin/${BRANCH}..."
    git fetch origin main
    if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
        git fetch origin "$BRANCH"
    fi

    main_version=$(read_elasticsearch_version_from_ref "origin/main")
    if [[ -z "$main_version" ]]; then
        echo "ERROR: could not read elasticsearchVersion from origin/main gradle.properties" >&2
        exit 1
    fi
    echo "Current version on origin/main: ${main_version}"

    release_branch_exists=false
    release_branch_version=""
    if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
        release_branch_exists=true
        release_branch_version=$(read_elasticsearch_version_from_ref "origin/${BRANCH}")
        echo "Release branch origin/${BRANCH} exists at version: ${release_branch_version:-unknown}"
    else
        echo "Release branch origin/${BRANCH} does not exist yet"
    fi

    minor_validate_args=(
        "$PYTHON" "$VALIDATION_PY" validate-minor-freeze
        --main-version "$main_version"
        --new "$NEW_VERSION"
        --branch "$BRANCH"
    )
    if [[ "$release_branch_exists" == "true" ]]; then
        minor_validate_args+=(--release-branch-exists --release-branch-version "$release_branch_version")
    fi
    if ! "${minor_validate_args[@]}"; then
        exit 1
    fi

    MAIN_NEW_VERSION=$("$PYTHON" "$VALIDATION_PY" derive-main-new-version --new "$NEW_VERSION")
    version_bump_set_buildkite_meta "ml_cpp_version_bump_main_new_version" "$MAIN_NEW_VERSION"
    echo "Derived MAIN_NEW_VERSION for main bump: ${MAIN_NEW_VERSION}"

    branch_needed=true
    if [[ "$release_branch_exists" == "true" && "$release_branch_version" == "$NEW_VERSION" ]]; then
        branch_needed=false
    fi
    version_bump_set_buildkite_meta "ml_cpp_minor_branch_needed" "$([[ "$branch_needed" == "true" ]] && echo true || echo false)"

    main_bump_needed=true
    main_trim=$(echo "$main_version" | tr -d '[:space:]')
    main_new_trim=$(echo "$MAIN_NEW_VERSION" | tr -d '[:space:]')
    if [[ "$main_trim" == "$main_new_trim" ]]; then
        main_bump_needed=false
    fi
    if [[ "$BRANCH" == testing-* ]]; then
        main_bump_needed=false
        echo "Sandbox branch ${BRANCH} — main bump will be skipped"
    fi
    version_bump_set_buildkite_meta "ml_cpp_main_bump_needed" "$([[ "$main_bump_needed" == "true" ]] && echo true || echo false)"

    if [[ "$branch_needed" == "false" && "$main_bump_needed" == "false" ]]; then
        version_bump_set_noop_meta true
        echo "OK: release branch and main bump already complete — follow-up steps will no-op."
    else
        version_bump_set_noop_meta false
    fi
    exit 0
fi

if [[ "$WORKFLOW" != "patch" ]]; then
    echo "ERROR: WORKFLOW must be \"patch\" or \"minor\", got \"${WORKFLOW}\"" >&2
    exit 1
fi

echo "=== Version bump validation (patch) ==="
echo "WORKFLOW:     ${WORKFLOW}"
echo "NEW_VERSION:  ${NEW_VERSION}"
echo "BRANCH:       ${BRANCH}"

# Patch workflow: consecutive patch increment on the release branch named BRANCH.

echo "Fetching origin/${BRANCH}..."
git fetch origin "$BRANCH"

if ! git cat-file -e FETCH_HEAD:gradle.properties 2>/dev/null; then
    echo "ERROR: gradle.properties missing at FETCH_HEAD (origin/${BRANCH})" >&2
    exit 1
fi

# Allow empty result: with pipefail, grep exits 1 when there is no match, which
# would abort the substitution before the explicit empty check below.
CURRENT_VERSION=$(
    git show FETCH_HEAD:gradle.properties | grep '^elasticsearchVersion=' | head -1 | cut -d= -f2 | tr -d '[:space:]' || true
)

if [[ -z "$CURRENT_VERSION" ]]; then
    echo "ERROR: could not read elasticsearchVersion from origin/${BRANCH} gradle.properties" >&2
    exit 1
fi

echo "Current version on origin/${BRANCH}: ${CURRENT_VERSION}"

if ! "$PYTHON" "$VALIDATION_PY" validate-and-report \
    --current "$CURRENT_VERSION" \
    --new "$NEW_VERSION" \
    --branch "$BRANCH"
then
    exit 1
fi

# Compare trimmed forms for no-op meta-data (gradle value is already space-stripped).
cur_trim=$(echo "$CURRENT_VERSION" | tr -d '[:space:]')
new_trim=$(echo "$NEW_VERSION" | tr -d '[:space:]')
if [[ "$cur_trim" == "$new_trim" ]]; then
    version_bump_set_noop_meta true
else
    version_bump_set_noop_meta false
fi
