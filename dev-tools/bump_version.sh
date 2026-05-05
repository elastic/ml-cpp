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
# step. After git pull, we re-run the same patch rules against the branch tip so a
# race (another bump / manual commit) cannot downgrade elasticsearchVersion.
#
# Updates elasticsearchVersion in gradle.properties to NEW_VERSION on BRANCH,
# commits, and pushes. Does not modify .backportrc.json (reserved for future
# release automation).
#
# Set DRY_RUN=true to perform all steps except git push.
#
# Follows the same pattern as the Elasticsearch repo's automated
# Lucene snapshot updates (.buildkite/scripts/lucene-snapshot/).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VALIDATION_PY="${SCRIPT_DIR}/version_bump_validation.py"

: "${NEW_VERSION:?NEW_VERSION must be set}"
: "${BRANCH:?BRANCH must be set}"
DRY_RUN="${DRY_RUN:-false}"

GRADLE_PROPS="gradle.properties"

if [ "$DRY_RUN" = "true" ]; then
    echo "=== DRY RUN MODE — will not push ==="
fi

git_push() {
    local target_branch="$1"
    if [ "$DRY_RUN" = "true" ]; then
        echo "  [DRY RUN] Would push $target_branch"
    else
        git push origin "$target_branch"
        echo "  Pushed $target_branch"
    fi
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

bump_version_on_branch() {
    local target_branch="$1"
    local target_version="$2"

    git checkout "$target_branch"
    git pull --ff-only origin "$target_branch"

    local current_version
    # pipefail: grep exits 1 when there is no match — do not abort before the
    # empty check below.
    current_version=$(
        grep '^elasticsearchVersion=' "$GRADLE_PROPS" | head -1 | cut -d= -f2 | tr -d '[:space:]' || true
    )
    if [[ -z "$current_version" ]]; then
        echo "ERROR: could not read elasticsearchVersion from ${GRADLE_PROPS} on ${target_branch}" >&2
        exit 1
    fi

    # Revalidate against post-pull branch tip (race with other bumps / manual edits).
    if ! "$PYTHON" "$VALIDATION_PY" validate \
        --current "$current_version" \
        --new "$target_version" \
        --branch "$target_branch"
    then
        echo "ERROR: version bump does not match branch tip after pull (current=${current_version}, target=${target_version})." >&2
        echo "Refusing to rewrite elasticsearchVersion — resolve manually if another automation advanced the branch." >&2
        exit 1
    fi

    if [ "$current_version" = "$target_version" ]; then
        echo "Version on $target_branch is already $target_version — nothing to do"
        return 0
    fi

    echo "Bumping version on $target_branch: $current_version → $target_version"
    sed_inplace "s/^elasticsearchVersion=.*/elasticsearchVersion=${target_version}/" "$GRADLE_PROPS"

    if ! grep -q "^elasticsearchVersion=${target_version}$" "$GRADLE_PROPS"; then
        echo "ERROR: version update verification failed on $target_branch"
        grep 'elasticsearchVersion' "$GRADLE_PROPS"
        exit 1
    fi

    if git diff-index --quiet HEAD --; then
        echo "No changes to commit on $target_branch (file unchanged after sed)"
        return 0
    fi

    configure_git
    git add "$GRADLE_PROPS"
    git commit -m "[ML] Bump version to ${target_version}"
    git_push "$target_branch"
}

echo "=== Patch version bump: $BRANCH → $NEW_VERSION ==="
bump_version_on_branch "$BRANCH" "$NEW_VERSION"

if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "=== DRY RUN SUMMARY ==="
    echo "Branch:   $BRANCH"
    echo "Version:  $NEW_VERSION"
    echo "Recent commits:"
    git log --oneline -3
fi
