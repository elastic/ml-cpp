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
# Second phase of the ml-cpp version-bump pipeline (after validate). Buildkite step
# `if` cannot read build meta-data, so we gate follow-up steps by reading
# ml_cpp_version_bump_noop here and uploading phase-2 YAML only when a bump is needed.

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

if [[ "${DRY_RUN:-}" == "true" ]]; then
    echo "DRY_RUN=true — not scheduling version-bump follow-up steps."
    exit 0
fi

if ! command -v buildkite-agent >/dev/null 2>&1; then
    echo "ERROR: buildkite-agent not found; cannot upload phase-2 pipeline." >&2
    exit 1
fi

noop=$(buildkite-agent meta-data get "ml_cpp_version_bump_noop" 2>/dev/null || echo "false")
if [[ "${noop}" == "true" ]]; then
    echo "ml_cpp_version_bump_noop=true — branch already at NEW_VERSION; skipping follow-up steps."
    exit 0
fi

exec python3 .buildkite/job-version-bump-phase2.json.py | buildkite-agent pipeline upload
