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
# Verifies that the current checkout can authenticate for git push to origin,
# without updating any remote refs (uses "git push --dry-run"), and (TEMPORARY)
# that an empty commit can be created and undone — same constraints as bump_version.sh.
#
# Intended for the ml-cpp-version-bump Buildkite pipeline (same agent + remotes
# as dev-tools/bump_version.sh). Uses a disposable ref name under refs/heads/ci/
# so it does not collide with release branches.
#
# Environment:
#   BUILDKITE_BUILD_NUMBER — used to uniquify the probe ref (defaults to "local"
#     when unset, e.g. manual runs outside Buildkite).
#   GIT_REMOTE — remote name (default: origin).

set -euo pipefail

REMOTE="${GIT_REMOTE:-origin}"
BUILD_NUM="${BUILDKITE_BUILD_NUMBER:-local}"
PROBE_REF="refs/heads/ci/ml-cpp-bump-push-probe-${BUILD_NUM}"

echo "=== Git push auth probe (dry-run; no remote refs updated) ==="
echo "Remote: ${REMOTE}"
echo "Local HEAD: $(git rev-parse HEAD)"
echo "Probe refspec: HEAD:${PROBE_REF}"
git remote -v

if ! git push --dry-run "${REMOTE}" "HEAD:${PROBE_REF}"; then
    echo "ERROR: git push --dry-run failed — check credentials and GitHub permissions for ${REMOTE}." >&2
    exit 1
fi

echo "OK: git push --dry-run succeeded for ${REMOTE}."

# TEMPORARY — remove once CI git identity + commit permissions are confirmed end-to-end.
# Mirrors configure_git in dev-tools/bump_version.sh (empty commit, then restore HEAD).
echo ""
echo "=== TEMPORARY: Git commit probe (empty commit, then reset) ==="
git config user.name elasticsearchmachine
git config user.email 'infra-root+elasticsearchmachine@elastic.co'
PRE_HEAD="$(git rev-parse HEAD)"
if ! git commit --allow-empty -m "[CI] Temporary empty commit probe"; then
    echo "ERROR: git commit failed — check git identity and repo permissions." >&2
    exit 1
fi
if ! git reset --hard "${PRE_HEAD}"; then
    echo "ERROR: git reset --hard failed after probe commit — workspace may be dirty." >&2
    exit 1
fi
echo "OK: git commit succeeded; HEAD restored to ${PRE_HEAD}."
