#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Generate an SBOM of the shipped third-party libraries and test it with Snyk.
# SNYK_TOKEN is injected by .buildkite/hooks/post-checkout for this step key.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}"

SNYK_ORG="${SNYK_ORG:-machine-learning-48e}"
SNYK_SEVERITY_THRESHOLD="${SNYK_SEVERITY_THRESHOLD:-high}"
SBOM_FILE="${SBOM_FILE:-sbom.cdx.json}"

python3 dev-tools/generate_sbom.py --output "${SBOM_FILE}"

if [ -z "${SNYK_TOKEN:-}" ]; then
    # Fail loudly rather than reporting a green scan that never ran. The step is
    # soft_fail, so this shows up as a warning and does not block the build.
    echo "ERROR: SNYK_TOKEN is not set; the Snyk scan did not run." >&2
    echo "Expected the post-checkout hook to read it from Vault for step key snyk_sbom_scan." >&2
    exit 1
fi

# Install the Snyk CLI and verify the published checksum before running it.
SNYK_BIN="$(mktemp -d)/snyk"
curl -fsSL -o "${SNYK_BIN}" https://static.snyk.io/cli/latest/snyk-linux
EXPECTED_SHA="$(curl -fsSL https://static.snyk.io/cli/latest/snyk-linux.sha256 | awk '{print $1}')"
echo "${EXPECTED_SHA}  ${SNYK_BIN}" | sha256sum -c -
chmod +x "${SNYK_BIN}"

# Non-zero exit when vulnerabilities at or above the threshold are found.
"${SNYK_BIN}" sbom test \
    --file="${SBOM_FILE}" \
    --org="${SNYK_ORG}" \
    --severity-threshold="${SNYK_SEVERITY_THRESHOLD}"
