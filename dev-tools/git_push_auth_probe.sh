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
# Verifies that the checkout can authenticate for git push to origin, without
# updating any remote refs (uses "git push --dry-run").
#
# Primary path (Buildkite sets BRANCH): dry-run FETCH_HEAD:refs/heads/${BRANCH}
# after fetching origin/${BRANCH} — same ref as dev-tools/bump_version.sh, so
# branch protection / rulesets on the release branch are exercised (not only a
# disposable ci/ ref).
#
# Fallback when BRANCH is unset (optional local use): disposable refs/heads/ci/
# — weaker (does not prove permission on the real release branch).
#
# Intended for the ml-cpp-version-bump Buildkite pipeline (same agent + remotes
# as dev-tools/bump_version.sh).
#
# Environment:
#   BRANCH — release branch (e.g. 9.5). Required when BUILDKITE=true (protected-ref
#     probe); optional locally (falls back to refs/heads/ci/...).
#   BUILDKITE_BUILD_NUMBER — used to uniquify the fallback ci/ probe ref
#     (defaults to "local" when unset).
#   GIT_REMOTE — remote name (default: origin).

set -euo pipefail

REMOTE="${GIT_REMOTE:-origin}"
BUILD_NUM="${BUILDKITE_BUILD_NUMBER:-local}"
# Collapse whitespace-only BRANCH (invalid for fetch / refspec).
BRANCH_TRIMMED="${BRANCH:-}"
BRANCH_TRIMMED="${BRANCH_TRIMMED// }"

echo "=== Git push auth probe (dry-run; no remote refs updated) ==="
echo "Remote: ${REMOTE}"
echo "Local HEAD: $(git rev-parse HEAD)"
git remote -v

if [[ "${BUILDKITE:-}" == "true" ]] && [[ -z "${BRANCH_TRIMMED}" ]]; then
    echo "ERROR: BRANCH must be set in Buildkite so the probe can dry-run refs/heads/\${BRANCH}." >&2
    exit 1
fi

if [[ -n "${BRANCH_TRIMMED}" ]]; then
    echo "Protected-ref probe: BRANCH=${BRANCH_TRIMMED} (same ref as bump_version.sh push target)"
    echo "Fetching origin/${BRANCH_TRIMMED}..."
    git fetch origin "${BRANCH_TRIMMED}"
    echo "FETCH_HEAD: $(git rev-parse FETCH_HEAD)"
    REFSPEC="FETCH_HEAD:refs/heads/${BRANCH_TRIMMED}"
else
    echo "WARNING: BRANCH unset — using disposable refs/heads/ci/ probe only (weaker; see script header)." >&2
    PROBE_REF="refs/heads/ci/ml-cpp-bump-push-probe-${BUILD_NUM}"
    REFSPEC="HEAD:${PROBE_REF}"
    echo "Fallback probe refspec: ${REFSPEC}"
fi

if ! git push --dry-run "${REMOTE}" "${REFSPEC}"; then
    echo "ERROR: git push --dry-run failed — check credentials and GitHub permissions for ${REMOTE}." >&2
    exit 1
fi

echo "OK: git push --dry-run succeeded for ${REMOTE}."
