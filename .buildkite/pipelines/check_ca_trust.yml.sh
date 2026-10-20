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
# Runs .buildkite/scripts/steps/check_ca_trust.sh in each Docker image (and on
# the aarch64 build host) that CI builds or tests with, to check whether its
# CA trust store still trusts the compromised eMudhra intermediates (InfoSec
# incident "Intrepid Falcon", https://emudhra.we-have-a-problem.com/).
#
# These run after the post-checkout hook, which applies the runtime mitigation
# (distrust_emudhra_cas.sh), so a PASS means the mitigation took effect.  The
# steps are soft-fail: a FAIL shows as a warning against the image that
# needs its CA store updated, without blocking the PR.

CHECK=".buildkite/scripts/steps/check_ca_trust.sh"

# Emits a step that runs the check inside the given Docker image.
image_step() {
    local key="$1" label="$2" image="$3"
    cat <<STEP
  - label: "CA trust check: ${label}"
    key: "check_ca_trust_${key}"
    command: "${CHECK}"
    soft_fail: true
    timeout_in_minutes: 10
    agents:
      image: "${image}"
STEP
}

cat <<EOL
steps:
EOL

image_step linux_build "Linux build image (x86_64)" "docker.elastic.co/ml-dev/ml-linux-build:34"
image_step pytorch_dependency "PyTorch dependency image" "docker.elastic.co/ml-dev/ml-linux-dependency-build:pytorch_latest"
image_step check_style "check-style image" "docker.elastic.co/ml-dev/ml-check-style:2"
image_step jdk17 "JDK 17 DRA image" "docker.elastic.co/ml-dev/ml_cpp_linux_x86_64_jdk17:3"
image_step python "python:3.11-bookworm image" "python:3.11-bookworm"

# aarch64 builds run on an AlmaLinux 8 VM host, and tests run inside the
# aarch64 native build image via "docker run", so check both.
cat <<EOL
  - label: "CA trust check: Linux aarch64 build host and native build image"
    key: "check_ca_trust_linux_aarch64"
    command:
      - "${CHECK}"
      - "docker pull docker.elastic.co/ml-dev/ml-linux-aarch64-native-build:17"
      - "docker run --rm -v \"\$(pwd)/.buildkite/scripts:/ci:ro\" docker.elastic.co/ml-dev/ml-linux-aarch64-native-build:17 bash -c 'bash /ci/common/distrust_emudhra_cas.sh; bash /ci/steps/check_ca_trust.sh'"
    soft_fail: true
    timeout_in_minutes: 15
    agents:
      provider: "aws"
      instanceType: "m6g.2xlarge"
      imagePrefix: "core-almalinux-8-aarch64"
      diskSizeGb: "100"
      diskName: "/dev/xvda"
EOL
