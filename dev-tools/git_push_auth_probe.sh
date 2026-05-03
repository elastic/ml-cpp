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
# without updating any remote refs (uses "git push --dry-run").
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

# TEMPORARY (version-bump PR): inspect GitHub App metadata; remove before merge.
echo "=== TEMPORARY: elastic-vault-github-plugin-prod — reported .permissions ==="
if [ -n "${VAULT_GITHUB_TOKEN:-}" ]; then
    if command -v gh >/dev/null 2>&1; then
        GH_TOKEN="${VAULT_GITHUB_TOKEN}" gh api /apps/elastic-vault-github-plugin-prod --jq '.permissions' ||
            echo "WARNING: gh api app permissions query failed (non-fatal)." >&2
    else
        echo "NOTE: gh not in PATH; using curl for the same REST endpoint." >&2
        curl -sS -H "Authorization: Bearer ${VAULT_GITHUB_TOKEN}" \
            -H "Accept: application/vnd.github+json" \
            "https://api.github.com/apps/elastic-vault-github-plugin-prod" |
            python3 -c 'import json, sys; print(json.dumps(json.load(sys.stdin).get("permissions"), indent=2))' ||
            echo "WARNING: curl/python app permissions query failed (non-fatal)." >&2
    fi
else
    echo "WARNING: VAULT_GITHUB_TOKEN unset; skipping app permissions diagnostic." >&2
fi

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
