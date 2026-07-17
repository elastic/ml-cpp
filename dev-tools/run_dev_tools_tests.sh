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
# Runs dev-tools/unittest via pytest (see dev-tools/test-requirements.txt).
# Does not install packages — pytest must already be importable (venv, image, or
# a prior pip install). PR CI runs .buildkite/scripts/steps/dev_tools_pytest.sh,
# which performs one pip install per job from dev-tools/test-requirements.txt
# (PyPI); a future improvement is baking deps into the CI image to reduce that
# dependency.
#
# Usage (from repository root):
#   ./dev-tools/run_dev_tools_tests.sh
#   ./dev-tools/run_dev_tools_tests.sh -q --tb=short

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || true
if [[ -z "${ROOT}" ]]; then
    echo "ERROR: not inside a git repository (git rev-parse --show-toplevel failed)." >&2
    echo "Run this script from a checkout of ml-cpp." >&2
    exit 1
fi
cd "${ROOT}/dev-tools"

if ! python3 -m pytest --version >/dev/null 2>&1; then
    echo "pytest is not installed in the current environment." >&2
    echo "Install dev-tools test dependencies first, for example:" >&2
    echo "  python3 -m pip install -r ${ROOT}/dev-tools/test-requirements.txt" >&2
    echo "On Buildkite, .buildkite/scripts/steps/dev_tools_pytest.sh installs this before invoking this script." >&2
    exit 1
fi

exec python3 -m pytest -c pytest.ini "$@"
