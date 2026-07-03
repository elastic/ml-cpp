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
# Minor feature freeze — Leg B: bump main to the next minor and update .backportrc.json.
#
# NEW_VERSION is the release-branch version (e.g. 9.5.0). main is bumped to
# MAIN_NEW_VERSION derived as minor+1 (e.g. 9.6.0). Opens a PR into main.
#
# Environment: same as dev-tools/bump_version.sh plus:
#   BRANCH — new release branch name (for .backportrc.json only)
#
# Buildkite meta-data:
#   ml_cpp_main_bump_changed — true when a main-bump PR was opened (or DRY_RUN simulates)
#   ml_cpp_version_bump_changed — same as ml_cpp_main_bump_changed (DRA / Slack compat)
#   ml_cpp_version_bump_main_new_version — MAIN_NEW_VERSION for DRA wait

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=version_bump_lib.sh
source "${SCRIPT_DIR}/version_bump_lib.sh"

PYTHON="${PYTHON:-python3}"
VALIDATION_PY="${SCRIPT_DIR}/version_bump_validation.py"
UPDATE_BACKPORTRC_PY="${SCRIPT_DIR}/update_backportrc.py"
CREATE_PR_SH="${SCRIPT_DIR}/create_github_pull_request.sh"

: "${NEW_VERSION:?NEW_VERSION must be set}"
: "${BRANCH:?BRANCH must be set}"

NEW_VERSION="$(version_bump_trim_value "${NEW_VERSION}")"
BRANCH="$(version_bump_trim_value "${BRANCH}")"
DRY_RUN="${DRY_RUN:-false}"
TARGET_BRANCH="main"

GRADLE_PROPS="gradle.properties"
BACKPORTRC=".backportrc.json"

if [ "$DRY_RUN" = "true" ]; then
    echo "=== DRY RUN MODE — will not push or create PR ==="
fi

version_bump_set_main_bump_changed() {
    local changed="$1"
    version_bump_set_buildkite_meta "ml_cpp_main_bump_changed" "$changed"
    version_bump_set_buildkite_meta_changed "$changed"
}

if [[ "$BRANCH" == testing-* ]]; then
    echo "Sandbox branch ${BRANCH} — skipping main bump and .backportrc.json update"
    version_bump_set_main_bump_changed false
    version_bump_set_buildkite_meta "ml_cpp_main_bump_needed" "false"
    exit 0
fi

MAIN_NEW_VERSION=$("$PYTHON" "$VALIDATION_PY" derive-main-new-version --new "$NEW_VERSION")
version_bump_set_buildkite_meta "ml_cpp_version_bump_main_new_version" "$MAIN_NEW_VERSION"

main_bump_topic_branch_name() {
    local tb="ci/ml-cpp-minor-freeze-main-${MAIN_NEW_VERSION}"
    if [[ -n "${BUILDKITE_BUILD_NUMBER:-}" ]]; then
        tb="${tb}-bk${BUILDKITE_BUILD_NUMBER}"
    fi
    echo "$tb"
}

echo "=== Minor freeze Leg B: bump ${TARGET_BRANCH} ${NEW_VERSION} → ${MAIN_NEW_VERSION} ==="
version_bump_set_main_bump_changed false

git fetch origin "$TARGET_BRANCH"

current_version=$(read_elasticsearch_version_from_ref "origin/${TARGET_BRANCH}")
if [[ -z "$current_version" ]]; then
    echo "ERROR: could not read elasticsearchVersion from origin/${TARGET_BRANCH}" >&2
    exit 1
fi

if ! "$PYTHON" "$VALIDATION_PY" validate-main-minor-bump \
    --current "$current_version" \
    --main-new-version "$MAIN_NEW_VERSION" \
    --release-branch-version "$NEW_VERSION"
then
    exit 1
fi

topic_branch=$(main_bump_topic_branch_name)
git checkout -B "$topic_branch" "origin/${TARGET_BRANCH}"

current_version=$(read_elasticsearch_version_from_file "$GRADLE_PROPS")

if [[ "$current_version" != "$MAIN_NEW_VERSION" ]]; then
    echo "Updating ${GRADLE_PROPS}: ${current_version} → ${MAIN_NEW_VERSION}"
    sed_inplace "s/^elasticsearchVersion=.*/elasticsearchVersion=${MAIN_NEW_VERSION}/" "$GRADLE_PROPS"
fi

if ! grep -q "^elasticsearchVersion=${MAIN_NEW_VERSION}$" "$GRADLE_PROPS"; then
    echo "ERROR: version update verification failed on ${topic_branch}" >&2
    grep 'elasticsearchVersion' "$GRADLE_PROPS" >&2
    exit 1
fi

if ! "$PYTHON" "$UPDATE_BACKPORTRC_PY" \
    --path "$BACKPORTRC" \
    --new-release-branch "$BRANCH" \
    --main-new-version "$MAIN_NEW_VERSION"
then
    exit 1
fi

if git diff-index --quiet HEAD --; then
    echo "main already at ${MAIN_NEW_VERSION} and .backportrc.json is up to date — nothing to do"
    version_bump_set_buildkite_meta "ml_cpp_main_bump_needed" "false"
    exit 0
fi

configure_git
git add "$GRADLE_PROPS" "$BACKPORTRC"
git commit -m "[ML] Bump version to ${MAIN_NEW_VERSION} (minor freeze)"

if [ "$DRY_RUN" = "true" ]; then
    echo "  [DRY RUN] Would push origin ${topic_branch} and open PR into ${TARGET_BRANCH}"
    version_bump_set_main_bump_changed true
    version_bump_set_buildkite_meta "ml_cpp_main_bump_needed" "true"
    exit 0
fi

git push -u origin "$topic_branch"
echo "  Pushed topic branch ${topic_branch}"

repo_slug=$(github_repo_slug) || exit 1

pr_body="$(cat <<EOF
Automated minor feature-freeze bump for \`${TARGET_BRANCH}\`.

| | |
| --- | --- |
| **Release branch** | \`${BRANCH}\` @ \`${NEW_VERSION}\` |
| **elasticsearchVersion on main** | \`${current_version}\` → \`${MAIN_NEW_VERSION}\` |
| **.backportrc.json** | Adds \`${BRANCH}\`; maps \`v${MAIN_NEW_VERSION}\` → \`main\` |

When merging is enabled (\`VERSION_BUMP_NO_MERGE\` not true): **auto-merge** if \`VERSION_BUMP_MERGE_AUTO=true\`.
EOF
)"

local -a pr_cmd=(
    "$CREATE_PR_SH"
    --repo "$repo_slug"
    --base "$TARGET_BRANCH"
    --head "$topic_branch"
    --title "[ML] Bump version to ${MAIN_NEW_VERSION} (minor freeze)"
    --body "$pr_body"
    --label "ci:skip-es-tests"
)
if [[ "${VERSION_BUMP_NO_MERGE:-}" != "true" ]]; then
    if [[ "${VERSION_BUMP_MERGE_AUTO:-}" == "true" ]]; then
        pr_cmd+=(--merge-auto)
    else
        pr_cmd+=(--merge)
    fi
fi
if [[ -n "${VERSION_BUMP_MERGE_METHOD:-}" ]]; then
    pr_cmd+=(--merge-method "${VERSION_BUMP_MERGE_METHOD}")
fi

pr_url=$("${pr_cmd[@]}")
echo "  Pull request: ${pr_url}"
version_bump_set_main_bump_changed true
version_bump_set_buildkite_meta "ml_cpp_main_bump_needed" "true"
version_bump_set_pr_url_meta "$pr_url"
