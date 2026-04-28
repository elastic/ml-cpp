#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Always validate against the published PyTorch Linux dependency image (same tag as
# Linux compile agents: torch + MKL under /usr/local/gcc133 per dev-tools/docker/pytorch_linux_image).
# Optional override for experiments: PYTORCH_ALLOWLIST_VALIDATION_IMAGE.
VALIDATION_IMAGE="${PYTORCH_ALLOWLIST_VALIDATION_IMAGE:-docker.elastic.co/ml-dev/ml-linux-dependency-build:pytorch_latest}"

cat <<EOL
steps:
  - label: "Validate PyTorch allowlist :torch:"
    key: "validate_pytorch_allowlist"
    timeout_in_minutes: 60
    env:
        HF_HUB_DISABLE_XET: "1"
    command:
        - "if [ ! -f dev-tools/extract_model_ops/validate_allowlist.py ]; then echo 'validate_allowlist.py not found, skipping'; exit 0; fi"
        - "python3 -c \"import torch; print(f'PyTorch version: {torch.__version__}')\""
        - "grep -v '^torch==' dev-tools/extract_model_ops/requirements.txt | pip3 install -r /dev/stdin"
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

cat <<EOL
    allow_dependency_failure: true
    agents:
      image: "${VALIDATION_IMAGE}"
      memory: "32G"
      ephemeralStorage: "30G"
    notify:
      - github_commit_status:
          context: "Validate PyTorch allowlist"
EOL
