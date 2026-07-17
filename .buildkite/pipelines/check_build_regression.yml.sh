#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

cat <<'EOL'
steps:
  - label: "Check build timing regressions :chart_with_downwards_trend:"
    key: "check_build_regression"
    command:
        - "python3 dev-tools/check_build_regression.py --annotate"
EOL

# Emit depends_on dynamically — ML_BUILD_STEP_KEYS is a comma-separated
# list of step keys set by the pipeline generator.  Only keys that
# actually exist in this build are included, avoiding Buildkite errors
# when a platform is not built.
if [ -n "${ML_BUILD_STEP_KEYS:-}" ]; then
    echo '    depends_on:'
    IFS=',' read -ra STEP_KEYS <<< "$ML_BUILD_STEP_KEYS"
    for key in "${STEP_KEYS[@]}"; do
        echo "        - \"${key}\""
    done
fi

cat <<'EOL'
    allow_dependency_failure: true
    soft_fail: true
    agents:
      image: "python:3"
EOL
