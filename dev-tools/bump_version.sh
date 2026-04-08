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
# Automated version bump script for the release-eng pipeline.
#
# Updates elasticsearchVersion in gradle.properties to $NEW_VERSION,
# commits, and pushes directly to the target branch.  Designed to be
# called from a Buildkite step with NEW_VERSION and BRANCH set.
#
# Set DRY_RUN=true to perform all steps except the final git push.
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

# Ensure we're on the correct branch and up to date
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

CURRENT_VERSION=$(grep '^elasticsearchVersion=' "$GRADLE_PROPS" | cut -d= -f2)
if [ "$CURRENT_VERSION" = "$NEW_VERSION" ]; then
    echo "Version is already $NEW_VERSION — nothing to do"
    exit 0
fi

echo "Bumping version: $CURRENT_VERSION → $NEW_VERSION"
# macOS sed requires -i '' while GNU sed requires -i without an argument
if sed --version >/dev/null 2>&1; then
    sed -i "s/^elasticsearchVersion=.*/elasticsearchVersion=${NEW_VERSION}/" "$GRADLE_PROPS"
else
    sed -i '' "s/^elasticsearchVersion=.*/elasticsearchVersion=${NEW_VERSION}/" "$GRADLE_PROPS"
fi

# Verify the substitution worked
if ! grep -q "^elasticsearchVersion=${NEW_VERSION}$" "$GRADLE_PROPS"; then
    echo "ERROR: version update verification failed"
    grep 'elasticsearchVersion' "$GRADLE_PROPS"
    exit 1
fi

# Check there's actually a diff (guards against no-op)
if git diff-index --quiet HEAD --; then
    echo "No changes to commit (file unchanged after sed)"
    exit 0
fi

git config user.name elasticsearchmachine
git config user.email 'infra-root+elasticsearchmachine@elastic.co'

git add "$GRADLE_PROPS"
git commit -m "[ML] Bump version to ${NEW_VERSION}"

if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "=== DRY RUN: commit created but NOT pushed ==="
    echo "Branch:  $BRANCH"
    echo "Version: $CURRENT_VERSION → $NEW_VERSION"
    echo "Commit:  $(git log -1 --format='%h %s')"
    echo "Author:  $(git log -1 --format='%an <%ae>')"
    echo ""
    echo "To inspect: git log -1 -p"
    echo "To undo:    git reset --soft HEAD~1 && git restore --staged $GRADLE_PROPS && git checkout -- $GRADLE_PROPS"
else
    git push origin "$BRANCH"
    echo "Version bumped to ${NEW_VERSION} on branch ${BRANCH}"
fi
