#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

set -euo pipefail

# Install Python dependencies (same stack as validate-changelogs.sh)
if ! command -v git &>/dev/null; then
  apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1
fi
python3 -m pip install --quiet --break-system-packages pyyaml jsonschema 2>/dev/null \
  || python3 -m pip install --quiet pyyaml jsonschema

echo "Running Python unit tests for dev-tools changelog scripts..."
python3 -m unittest discover -s dev-tools/unittest -p 'test_*.py' -v
