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
# the aarch64 build host and the macOS agent), and check_ca_trust.ps1 on the
# Windows agent, that CI builds or tests with, to check whether its
# CA trust store still trusts the compromised eMudhra intermediates (InfoSec
# incident "Intrepid Falcon", https://emudhra.we-have-a-problem.com/).
#
# The Linux steps run after the post-checkout hook, which applies the runtime mitigation
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
# The JDK 17 DRA steps run as uid 1000 (imageUID), where the runtime mitigation
# cannot apply, so the image itself has the roots distrusted at build time
# (version 5; earlier versions trust the compromised CAs).  :5 is used by both
# create_dra (buildkite-agent artifact transfer) and upload_dra_to_s3 (Vault
# credentials and a Gradle upload, which uses the Java trust store).
image_step jdk17 "JDK 17 DRA image" "docker.elastic.co/ml-dev/ml_cpp_linux_x86_64_jdk17:5"
# Not checked: ml-check-style only runs clang-format on the checkout (no network
# use) and has neither curl nor python3 to run the check with.
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

# macOS builds run natively on Orka VMs (there is no macOS image), using the
# same agent as .buildkite/pipelines/build_macos.json.py.  This only checks;
# there is no runtime mitigation for macOS yet.
cat <<EOL
  - label: "CA trust check: macOS aarch64 agent"
    key: "check_ca_trust_macos_aarch64"
    command: "${CHECK}"
    soft_fail: true
    timeout_in_minutes: 10
    agents:
      provider: "orka"
      image: "ml-macos-14-arm-003.orkasi"
EOL

# Windows builds run natively on GCP VMs, using the same agent as
# .buildkite/pipelines/build_windows.json.py.  This only checks; there is no
# runtime mitigation for Windows yet.
cat <<'EOL'
  - label: "CA trust check: Windows x86_64 agent"
    key: "check_ca_trust_windows_x86_64"
    command: '& .buildkite\scripts\steps\check_ca_trust.ps1'
    soft_fail: true
    timeout_in_minutes: 10
    agents:
      provider: "gcp"
      machineType: "c2-standard-16"
      minCpuPlatform: "Intel Cascade Lake"
      image: "family/ml-cpp-5-windows-2022"
EOL
