#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Use an image that has Python 3.12, source-built torch, and MKL under
# /usr/local/gcc133 so `import torch` matches ml-cpp's libtorch linkage.
#
# Child pipelines (e.g. PyTorch Docker nightly via build_pytorch_docker_image.yml.sh)
# set DOCKER_IMAGE to ml-linux-dependency-build:pytorch_latest for *compile* agents.
# That image does not ship MKL next to torch; reusing it here reproduces
# libmkl_intel_lp64.so.2 errors. Only honour DOCKER_IMAGE when it is a ml-linux-build
# image; otherwise default to the published ml-linux-build tag.
DEFAULT_VALIDATION_IMAGE="docker.elastic.co/ml-dev/ml-linux-build:34"
if [[ -n "${DOCKER_IMAGE:-}" && "${DOCKER_IMAGE}" == *ml-linux-build* ]]; then
  VALIDATION_IMAGE="${DOCKER_IMAGE}"
else
  VALIDATION_IMAGE="${DEFAULT_VALIDATION_IMAGE}"
fi

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
