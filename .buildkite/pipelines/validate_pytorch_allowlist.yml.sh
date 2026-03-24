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
  - label: "Validate PyTorch allowlist :torch:"
    key: "validate_pytorch_allowlist"
    command:
        - "cmake -DSOURCE_DIR=$(pwd) -DVALIDATE_CONFIG=$(pwd)/dev-tools/extract_model_ops/validation_models.json -DVALIDATE_PT_DIR=$(pwd)/dev-tools/extract_model_ops/es_it_models -DVALIDATE_VERBOSE=TRUE -P cmake/run-validation.cmake"
EOL

# Depend on the build steps so validation doesn't start before the
# pipeline is fully generated.
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
