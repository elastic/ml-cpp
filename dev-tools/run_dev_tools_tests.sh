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
# Install pytest (see dev-tools/test-requirements.txt) and run dev-tools/unittest.
#
# Usage (from repository root):
#   ./dev-tools/run_dev_tools_tests.sh
#   ./dev-tools/run_dev_tools_tests.sh -q --tb=short

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}/dev-tools"

python3 -m pip install -q -r test-requirements.txt
exec python3 -m pytest -c pytest.ini "$@"
