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
# Gate for the patch workflow's "Fetch DRA Artifacts" step. dev-tools/bump_version.sh
# sets ml_cpp_version_bump_changed=false when it opens no PR (branch already at
# NEW_VERSION, or the file edit produced no diff) — in that case there is nothing
# new for the plugin to wait on, so skip scheduling it rather than polling DRA
# manifests for up to 480 minutes across 2 retries on a version that will never
# change.
#
# Buildkite step `if` cannot read build meta-data, so gating happens here instead
# (same pattern as dev-tools/version_bump_upload_phase2.sh).

set -euo pipefail

if [[ -n "${BUILDKITE_BUILD_CHECKOUT_PATH:-}" ]]; then
    cd "${BUILDKITE_BUILD_CHECKOUT_PATH}"
else
    ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -z "${ROOT}" ]]; then
        echo "ERROR: set BUILDKITE_BUILD_CHECKOUT_PATH or run from a git checkout" >&2
        exit 1
    fi
    cd "${ROOT}"
fi

if ! command -v buildkite-agent >/dev/null 2>&1; then
    echo "ERROR: buildkite-agent not found; cannot check version-bump status." >&2
    exit 1
fi

changed=$(buildkite-agent meta-data get "ml_cpp_version_bump_changed" 2>/dev/null || echo "false")
if [[ "${changed}" != "true" ]]; then
    echo "ml_cpp_version_bump_changed=${changed} — no PR opened; skipping DRA wait." >&2
    exit 0
fi

python3 .buildkite/job-dra-wait.json.py | buildkite-agent pipeline upload
