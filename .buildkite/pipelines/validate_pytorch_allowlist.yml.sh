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
    timeout_in_minutes: 60
    command:
        - "if [ ! -f dev-tools/extract_model_ops/validate_allowlist.py ]; then echo 'validate_allowlist.py not found, skipping'; exit 0; fi"
        - "pip install -r dev-tools/extract_model_ops/requirements.txt"
        - "python3 dev-tools/extract_model_ops/validate_allowlist.py --config dev-tools/extract_model_ops/validation_models.json --pt-dir dev-tools/extract_model_ops/es_it_models --verbose"
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
    agents:
      image: "python:3.12"
      memory: "32G"
      ephemeralStorage: "30G"
    notify:
      - github_commit_status:
          context: "Validate PyTorch allowlist"
EOL
