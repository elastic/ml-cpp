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
# Supports two workflows:
#   WORKFLOW=patch (default)
#     Updates elasticsearchVersion in gradle.properties to $NEW_VERSION
#     on $BRANCH, commits, and pushes.
#
#   WORKFLOW=minor
#     1. Creates a new minor branch from $BRANCH (e.g., 9.4 from main)
#        inheriting the current version.
#     2. Bumps $BRANCH to $NEW_VERSION (the next minor).
#     Both branches are pushed.
#
# Set DRY_RUN=true to perform all steps except git push.
#
# Follows the same pattern as the Elasticsearch repo's automated
# Lucene snapshot updates (.buildkite/scripts/lucene-snapshot/).

set -euo pipefail

: "${NEW_VERSION:?NEW_VERSION must be set}"
: "${BRANCH:?BRANCH must be set}"
WORKFLOW="${WORKFLOW:-patch}"
DRY_RUN="${DRY_RUN:-false}"

GRADLE_PROPS="gradle.properties"
BACKPORT_CONFIG=".backportrc.json"

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

    # Update .backportrc.json so the new version maps to main
    if [[ "$target_branch" == "main" && -f "$BACKPORT_CONFIG" ]]; then
        local escaped_version
        escaped_version=$(echo "$target_version" | sed 's/\./\\./g')
        echo "Updating backport config: v${target_version} → main"
        # Use python for a reliable cross-platform JSON-safe replacement
        python3 -c "
import json, re, sys
with open('$BACKPORT_CONFIG') as f:
    data = json.load(f)
mapping = data.get('branchLabelMapping', {})
new_mapping = {}
for k, v in mapping.items():
    if v == 'main' and re.match(r'\^v\d+\.\d+\.\d+\\\$', k):
        new_mapping['^v${target_version}\$'] = 'main'
    else:
        new_mapping[k] = v
data['branchLabelMapping'] = new_mapping
with open('$BACKPORT_CONFIG', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
" || echo "WARNING: could not update backport config — please check $BACKPORT_CONFIG manually"
    fi

    if git diff-index --quiet HEAD --; then
        echo "No changes to commit on $target_branch (file unchanged after sed)"
        return 0
    fi

    configure_git
    git add "$GRADLE_PROPS" "$BACKPORT_CONFIG"
    git commit -m "[ML] Bump version to ${target_version}"
    git_push "$target_branch"
}

# ---------------------------------------------------------------------------
# Patch workflow: bump version on the target branch
# ---------------------------------------------------------------------------
do_patch() {
    echo "=== Patch workflow: bump $BRANCH to $NEW_VERSION ==="
    bump_version_on_branch "$BRANCH" "$NEW_VERSION"
}

# ---------------------------------------------------------------------------
# Minor workflow: create minor branch, then bump upstream to next minor
# ---------------------------------------------------------------------------
do_minor() {
    echo "=== Minor workflow: create minor branch from $BRANCH, then bump to $NEW_VERSION ==="

    git checkout "$BRANCH"
    git pull --ff-only origin "$BRANCH"

    local current_version
    current_version=$(grep '^elasticsearchVersion=' "$GRADLE_PROPS" | cut -d= -f2)

    # Derive the minor branch name from current version (e.g., 9.4.0 → 9.4)
    local major minor
    major=$(echo "$current_version" | cut -d. -f1)
    minor=$(echo "$current_version" | cut -d. -f2)
    local minor_branch="${major}.${minor}"

    echo "Current version on $BRANCH: $current_version"
    echo "Minor branch to create: $minor_branch"
    echo "New version for $BRANCH: $NEW_VERSION"

    # Export minor branch info for downstream Buildkite steps
    if [ "${BUILDKITE:-false}" = "true" ]; then
        buildkite-agent meta-data set "MINOR_BRANCH" "$minor_branch"
        buildkite-agent meta-data set "MINOR_VERSION" "$current_version"
    fi
    export MINOR_BRANCH="$minor_branch"
    export MINOR_VERSION="$current_version"

    # Check if the minor branch already exists on the remote
    if git ls-remote --exit-code --heads origin "$minor_branch" >/dev/null 2>&1; then
        echo "Branch $minor_branch already exists on origin — skipping creation"
    else
        echo "Creating branch $minor_branch from $BRANCH..."
        git checkout -b "$minor_branch"
        configure_git
        git_push "$minor_branch"
        echo "Branch $minor_branch created with version $current_version"
    fi

    # Now bump the upstream branch to the new version
    bump_version_on_branch "$BRANCH" "$NEW_VERSION"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "$WORKFLOW" in
    patch)
        do_patch
        ;;
    minor)
        do_minor
        ;;
    *)
        echo "ERROR: unknown WORKFLOW '$WORKFLOW' (expected 'patch' or 'minor')"
        exit 1
        ;;
esac

if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "=== DRY RUN SUMMARY ==="
    echo "Workflow: $WORKFLOW"
    echo "Branch:   $BRANCH"
    echo "Version:  $NEW_VERSION"
    echo "Recent commits:"
    git log --oneline -3
fi
