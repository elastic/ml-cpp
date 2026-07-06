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
# Minor feature freeze — Leg A: create release branch from main (direct ref push).
#
# NEW_VERSION is the version expected on BRANCH (e.g. 9.5.0 on 9.5). main must
# already be at NEW_VERSION; this step does not change any version file.
#
# Environment:
#   NEW_VERSION, BRANCH — required (from release-eng; WORKFLOW=minor)
#   DRY_RUN — true to skip push
#
# Buildkite meta-data:
#   ml_cpp_minor_branch_created — true when branch was created or already OK
#   ml_cpp_minor_branch_needed — false when branch already exists at NEW_VERSION

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=version_bump_lib.sh
source "${SCRIPT_DIR}/version_bump_lib.sh"

PYTHON="${PYTHON:-python3}"
VALIDATION_PY="${SCRIPT_DIR}/version_bump_validation.py"

: "${NEW_VERSION:?NEW_VERSION must be set}"
: "${BRANCH:?BRANCH must be set}"

NEW_VERSION="$(version_bump_trim_value "${NEW_VERSION}")"
BRANCH="$(version_bump_trim_value "${BRANCH}")"
DRY_RUN="${DRY_RUN:-false}"
UPSTREAM_BRANCH="main"

if [ "$DRY_RUN" = "true" ]; then
    echo "=== DRY RUN MODE — will not push release branch ==="
fi

echo "=== Minor freeze Leg A: create release branch ${BRANCH} @ ${NEW_VERSION} ==="

git fetch origin "$UPSTREAM_BRANCH"

main_version=$(read_elasticsearch_version_from_ref "origin/${UPSTREAM_BRANCH}")
if [[ -z "$main_version" ]]; then
    echo "ERROR: could not read elasticsearchVersion from origin/${UPSTREAM_BRANCH}" >&2
    exit 1
fi

release_branch_exists=false
release_branch_version=""
if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
    release_branch_exists=true
    git fetch origin "$BRANCH"
    release_branch_version=$(read_elasticsearch_version_from_ref "origin/${BRANCH}")
fi

minor_validate_args=(
    --main-version "$main_version"
    --new "$NEW_VERSION"
    --branch "$BRANCH"
)
if [[ "$release_branch_exists" == "true" ]]; then
    minor_validate_args+=(--release-branch-exists --release-branch-version "$release_branch_version")
fi

if ! "$PYTHON" "$VALIDATION_PY" validate-minor-freeze "${minor_validate_args[@]}"
then
    exit 1
fi

if [[ "$release_branch_exists" == "true" ]]; then
    echo "Release branch origin/${BRANCH} already exists at ${NEW_VERSION} — nothing to do"
    version_bump_set_buildkite_meta "ml_cpp_minor_branch_created" "true"
    version_bump_set_buildkite_meta "ml_cpp_minor_branch_needed" "false"
    exit 0
fi

if [[ "$main_version" != "$NEW_VERSION" ]]; then
    echo "ERROR: origin/${UPSTREAM_BRANCH} is ${main_version}, expected ${NEW_VERSION} before branching" >&2
    exit 1
fi

if [ "$DRY_RUN" = "true" ]; then
    echo "  [DRY RUN] Would push origin/${UPSTREAM_BRANCH} -> refs/heads/${BRANCH}"
    version_bump_set_buildkite_meta "ml_cpp_minor_branch_created" "true"
    version_bump_set_buildkite_meta "ml_cpp_minor_branch_needed" "true"
    exit 0
fi

configure_git
git push origin "origin/${UPSTREAM_BRANCH}:refs/heads/${BRANCH}"
echo "  Created origin/${BRANCH} from origin/${UPSTREAM_BRANCH}"

git fetch origin "$BRANCH"
branch_version=$(read_elasticsearch_version_from_ref "origin/${BRANCH}")
if [[ "$branch_version" != "$NEW_VERSION" ]]; then
    echo "ERROR: origin/${BRANCH} version is ${branch_version}, expected ${NEW_VERSION}" >&2
    exit 1
fi

version_bump_set_buildkite_meta "ml_cpp_minor_branch_created" "true"
version_bump_set_buildkite_meta "ml_cpp_minor_branch_needed" "true"
echo "OK: release branch ${BRANCH} is at ${NEW_VERSION}"
