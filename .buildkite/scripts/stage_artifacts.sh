#!/usr/bin/env bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
# Stage DRA artifacts into artifacts/ for the elastic/dra-prep plugin.
#

set -euo pipefail

WORKFLOW="${DRA_WORKFLOW:?DRA_WORKFLOW is required}"

echo "--- :compression: Downloading ${WORKFLOW} artifacts from create_dra_artifacts step"
rm -rf build/distributions artifacts
mkdir -p build/distributions artifacts

buildkite-agent artifact download 'build/distributions/*.zip' . --step create_dra_artifacts
buildkite-agent artifact download 'build/distributions/*.csv' . --step create_dra_artifacts

echo "--- :package: Staging ${WORKFLOW} artifacts"
cp build/distributions/*.zip artifacts/
cp build/distributions/*.csv artifacts/

if ! ls artifacts/* 1>/dev/null 2>&1; then
  echo "ERROR: no ${WORKFLOW} artifacts found in artifacts/." >&2
  exit 1
fi

echo "Staged artifacts:"
ls -1 artifacts/
