#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
# Installs dev-tools pytest deps once per job (PyPI), then runs run_dev_tools_tests.sh.
# Keeping pip here isolates network/bootstrap from the test runner; baking deps into
# the agent image would avoid live PyPI on every PR if desired.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}"

if ! python3 -m pip install -q -r "${REPO_ROOT}/dev-tools/test-requirements.txt"; then
    echo "ERROR: pip install failed for dev-tools/test-requirements.txt (network or PyPI?)." >&2
    echo "Install manually on this agent, then re-run:" >&2
    echo "  python3 -m pip install -r ${REPO_ROOT}/dev-tools/test-requirements.txt" >&2
    exit 1
fi

exec "${REPO_ROOT}/dev-tools/run_dev_tools_tests.sh" -q --tb=short
