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
# Optionally merges after PR creation unless VERSION_BUMP_NO_MERGE=true:
#   VERSION_BUMP_MERGE_AUTO=true — gh pr merge --auto --squash (merge when checks pass; default for CI pipeline)
#   else — immediate gh pr merge --squash (legacy / escape hatch)
# Does not modify .backportrc.json (reserved for future release automation).
#
# Helpers used after `git checkout` of the release branch (validation + PR
# creation) are snapshotted from this script's directory first so an older
# branch tip cannot replace them mid-run.
#
# Environment:
#   NEW_VERSION, BRANCH — required
#   DRY_RUN — true to skip push and PR creation
#   BUILDKITE_BUILD_NUMBER — appended to topic branch name for uniqueness
#   VERSION_BUMP_TOPIC_BRANCH — optional override for topic branch name
#   VAULT_GITHUB_TOKEN / GH_TOKEN — auth for gh in CI (CI sets Vault token).
#   Local runs use `gh auth login`; GITHUB_TOKEN is ignored if set in the shell.
#   VERSION_BUMP_NO_MERGE — set to true to open PR only (no merge / auto-merge step)
#   VERSION_BUMP_MERGE_AUTO — true: enable GitHub auto-merge (--auto --squash); false/unset with merge: immediate squash
#   VERSION_BUMP_MERGE_METHOD — merge | squash | rebase (default: squash)
#   VERSION_BUMP_MERGE_ADMIN — true to pass gh pr merge --admin (needs repo bypass rights)
#   gh install (apk/tarball): dev-tools/ensure_github_cli.sh via create_github_pull_request.sh
#
# Buildkite (BUILDKITE=true): sets meta-data:
#   ml_cpp_version_bump_changed — true|false so the DRA wait step can skip when no PR was opened
#   ml_cpp_version_bump_pr_url — HTTPS URL of the opened PR (only set when non-empty;
#     Buildkite forbids empty meta-data values)
#
# Follows the same pattern as the Elasticsearch repo's automated
# Lucene snapshot updates (.buildkite/scripts/lucene-snapshot/).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=version_bump_lib.sh
source "${SCRIPT_DIR}/version_bump_lib.sh"

PYTHON="${PYTHON:-python3}"

# Snapshot before any release-branch checkout replaces worktree helpers.
HELPERS_DIR="$(version_bump_snapshot_helpers "$SCRIPT_DIR" \
    version_bump_validation.py \
    create_github_pull_request.sh \
    ensure_github_cli.sh)"
trap 'rm -rf "${HELPERS_DIR}"' EXIT
VALIDATION_PY="${HELPERS_DIR}/version_bump_validation.py"
CREATE_PR_SH="${HELPERS_DIR}/create_github_pull_request.sh"

: "${NEW_VERSION:?NEW_VERSION must be set}"
: "${BRANCH:?BRANCH must be set}"

# Normalise env (Buildkite / Windows agents may inject trailing CR or spaces).
NEW_VERSION="$(version_bump_trim_value "${NEW_VERSION}")"
BRANCH="$(version_bump_trim_value "${BRANCH}")"

DRY_RUN="${DRY_RUN:-false}"

GRADLE_PROPS="gradle.properties"

if [ "$DRY_RUN" = "true" ]; then
    echo "=== DRY RUN MODE — will not push or create PR ==="
fi

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

bump_version_via_pr() {
    local target_branch="$1"
    local target_version="$2"
    local topic_branch current_version repo_slug pr_url

    # Default: no DRA wait unless we open a PR (or DRY_RUN simulates one).
    version_bump_set_buildkite_meta_changed false

    topic_branch=$(topic_branch_name)

    git fetch origin "$target_branch"

    # Topic branch starts at release-branch tip (same tree validation uses).
    git checkout -B "$topic_branch" "origin/${target_branch}"

    current_version=$(read_elasticsearch_version_from_file "$GRADLE_PROPS")
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
        version_bump_set_buildkite_meta_changed true
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

When merging is enabled (\`VERSION_BUMP_NO_MERGE\` not true): **auto-merge** (\`gh pr merge --auto --squash\`) if \`VERSION_BUMP_MERGE_AUTO=true\`, otherwise **immediate squash** (\`gh pr merge --squash\`). Override merge style with \`VERSION_BUMP_MERGE_METHOD\`.
EOF
)"

    local -a pr_cmd=(
        "$CREATE_PR_SH"
        --repo "$repo_slug"
        --base "$target_branch"
        --head "$topic_branch"
        --title "[ML] Bump version to ${target_version}"
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
    version_bump_set_buildkite_meta_changed true
    version_bump_set_pr_url_meta "$pr_url"
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
