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
# Updates elasticsearchVersion in gradle.properties to NEW_VERSION on BRANCH,
# commits, and pushes. Does not modify .backportrc.json (reserved for a future
# main / minor bump automation change).
#
# Set DRY_RUN=true to perform all steps except git push.
#
# Follows the same pattern as the Elasticsearch repo's automated
# Lucene snapshot updates (.buildkite/scripts/lucene-snapshot/).

set -euo pipefail

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
    current_version=$(grep '^elasticsearchVersion=' "$GRADLE_PROPS" | cut -d= -f2)
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
