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
# Automated patch version bump for the release-eng pipeline.
#
# Parameter checks run in dev-tools/validate_version_bump_params.sh before this
# step. After git fetch, we re-run the same patch rules against the branch tip so a
# race (another bump / manual edits) cannot downgrade elasticsearchVersion.
#
# Creates a topic branch from origin/${BRANCH}, commits elasticsearchVersion in
# gradle.properties, pushes the topic branch to origin, and opens a GitHub pull
# request into ${BRANCH} via gh pr create (rulesets often disallow direct pushes).
# Optionally merges immediately with gh pr merge (unless VERSION_BUMP_NO_MERGE=true).
# Does not modify .backportrc.json (reserved for future release automation).
#
# Environment:
#   NEW_VERSION, BRANCH — required
#   DRY_RUN — true to skip push and PR creation
#   BUILDKITE_BUILD_NUMBER — appended to topic branch name for uniqueness
#   VERSION_BUMP_TOPIC_BRANCH — optional override for topic branch name
#   GITHUB_TOKEN / VAULT_GITHUB_TOKEN / GH_TOKEN — auth for gh (CI sets Vault token)
#   VERSION_BUMP_NO_MERGE — set to true to open PR only (no immediate gh pr merge)
#   VERSION_BUMP_MERGE_METHOD — merge | squash | rebase (default: squash)
#   gh install (apk/tarball): dev-tools/ensure_github_cli.sh via create_github_pull_request.sh
#
# Follows the same pattern as the Elasticsearch repo's automated
# Lucene snapshot updates (.buildkite/scripts/lucene-snapshot/).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VALIDATION_PY="${SCRIPT_DIR}/version_bump_validation.py"
CREATE_PR_SH="${SCRIPT_DIR}/create_github_pull_request.sh"

: "${NEW_VERSION:?NEW_VERSION must be set}"
: "${BRANCH:?BRANCH must be set}"
DRY_RUN="${DRY_RUN:-false}"

GRADLE_PROPS="gradle.properties"

if [ "$DRY_RUN" = "true" ]; then
    echo "=== DRY RUN MODE — will not push or create PR ==="
fi

# Parse elastic/ml-cpp from origin (https://github.com/elastic/ml-cpp.git or git@...)
github_repo_slug() {
    local url
    url=$(git remote get-url origin 2>/dev/null || true)
    if [[ "$url" =~ github\.com[:/]([^/]+)/([^/.]+)(\.git)?$ ]]; then
        echo "${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
        return 0
    fi
    echo "ERROR: could not parse owner/repo from git remote url: ${url:-empty}" >&2
    return 1
}

topic_branch_name() {
    local tb
    if [[ -n "${VERSION_BUMP_TOPIC_BRANCH:-}" ]]; then
        echo "${VERSION_BUMP_TOPIC_BRANCH}"
        return 0
    fi
    tb="ci/ml-cpp-version-bump-${BRANCH}-${NEW_VERSION}"
    if [[ -n "${BUILDKITE_BUILD_NUMBER:-}" ]]; then
        tb="${tb}-bk${BUILDKITE_BUILD_NUMBER}"
    fi
    echo "$tb"
}

sed_inplace() {
    if sed --version >/dev/null 2>&1; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

configure_git() {
    git config user.name elasticsearchmachine
    git config user.email 'infra-root+elasticsearchmachine@elastic.co'
}

bump_version_via_pr() {
    local target_branch="$1"
    local target_version="$2"
    local topic_branch current_version repo_slug pr_url

    topic_branch=$(topic_branch_name)

    git fetch origin "$target_branch"

    # Topic branch starts at release-branch tip (same tree validation uses).
    git checkout -B "$topic_branch" "origin/${target_branch}"

    current_version=$(
        grep '^elasticsearchVersion=' "$GRADLE_PROPS" | head -1 | cut -d= -f2 | tr -d '[:space:]' || true
    )
    if [[ -z "$current_version" ]]; then
        echo "ERROR: could not read elasticsearchVersion from ${GRADLE_PROPS} on origin/${target_branch}" >&2
        exit 1
    fi

    if ! "$PYTHON" "$VALIDATION_PY" validate \
        --current "$current_version" \
        --new "$target_version" \
        --branch "$target_branch"
    then
        echo "ERROR: version bump does not match branch tip after fetch (current=${current_version}, target=${target_version})." >&2
        echo "Refusing to rewrite elasticsearchVersion — resolve manually if another automation advanced the branch." >&2
        exit 1
    fi

    if [ "$current_version" = "$target_version" ]; then
        echo "Version on origin/${target_branch} is already ${target_version} — nothing to do"
        return 0
    fi

    echo "Bumping version via PR branch ${topic_branch}: ${current_version} → ${target_version} (base ${target_branch})"
    sed_inplace "s/^elasticsearchVersion=.*/elasticsearchVersion=${target_version}/" "$GRADLE_PROPS"

    if ! grep -q "^elasticsearchVersion=${target_version}$" "$GRADLE_PROPS"; then
        echo "ERROR: version update verification failed on ${topic_branch}"
        grep 'elasticsearchVersion' "$GRADLE_PROPS"
        exit 1
    fi

    if git diff-index --quiet HEAD --; then
        echo "No changes to commit (file unchanged after sed)"
        return 0
    fi

    configure_git
    git add "$GRADLE_PROPS"
    git commit -m "[ML] Bump version to ${target_version}"

    if [ "$DRY_RUN" = "true" ]; then
        echo "  [DRY RUN] Would push origin ${topic_branch} and open PR into ${target_branch}"
        return 0
    fi

    git push -u origin "$topic_branch"
    echo "  Pushed topic branch ${topic_branch}"

    repo_slug=$(github_repo_slug) || exit 1

    local pr_body
    pr_body="$(cat <<EOF
Automated patch version bump for branch \`${target_branch}\`.

| | |
| --- | --- |
| **elasticsearchVersion** | \`${current_version}\` → \`${target_version}\` |

Squash-merged immediately by the version-bump pipeline via \`gh pr merge --squash\` unless \`VERSION_BUMP_NO_MERGE=true\` (override style with \`VERSION_BUMP_MERGE_METHOD\`).
EOF
)"

    local -a pr_cmd=(
        "$CREATE_PR_SH"
        --repo "$repo_slug"
        --base "$target_branch"
        --head "$topic_branch"
        --title "[ML] Bump version to ${target_version}"
        --body "$pr_body"
    )
    if [[ "${VERSION_BUMP_NO_MERGE:-}" != "true" ]]; then
        pr_cmd+=(--merge)
    fi
    if [[ -n "${VERSION_BUMP_MERGE_METHOD:-}" ]]; then
        pr_cmd+=(--merge-method "${VERSION_BUMP_MERGE_METHOD}")
    fi

    pr_url=$("${pr_cmd[@]}")
    echo "  Pull request: ${pr_url}"
}

echo "=== Patch version bump (PR workflow): ${BRANCH} → ${NEW_VERSION} ==="
bump_version_via_pr "$BRANCH" "$NEW_VERSION"

if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "=== DRY RUN SUMMARY ==="
    echo "Branch:         $BRANCH"
    echo "Version:        $NEW_VERSION"
    echo "Topic branch:   $(topic_branch_name)"
    echo "Recent commits:"
    git log --oneline -3
fi
