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
# Runs check_ca_trust.sh in each CI Docker image, on the aarch64 build host and
# on the macOS agent, and check_ca_trust.ps1 on the Windows agent, to check
# whether they trust the compromised eMudhra CAs (InfoSec incident "Intrepid
# Falcon").  The steps run after the post-checkout mitigation, so PASS means
# it took effect.  Exit 1 (a store still trusts them) fails the build; exit 2
# (inconclusive, e.g. the test site is unreachable) only warns.  The aarch64
# test container, a known gap, is soft-fail until fixed.

CHECK=".buildkite/scripts/steps/check_ca_trust.sh"

SOFT_FAIL_INCONCLUSIVE='    soft_fail:
      - exit_status: 2'

AARCH64_AGENTS='    agents:
      provider: "aws"
      instanceType: "m6g.2xlarge"
      imagePrefix: "core-almalinux-8-aarch64"
      diskSizeGb: "100"
      diskName: "/dev/xvda"'

# Emits a step that runs the check inside the given Docker image, without the
# post-checkout mitigation, so it reports the image as-is.  A fifth argument
# "gap" soft-fails the step: a known gap until the image is rebuilt.
image_step() {
    local key="$1" label="$2" image="$3" uid="${4:-}" gap="${5:-}"
    local soft_fail="${SOFT_FAIL_INCONCLUSIVE}"
    [ "${gap}" = gap ] && soft_fail='    soft_fail: true'
    cat <<STEP
  - label: "CA trust check: ${label}"
    key: "check_ca_trust_${key}"
    command: "${CHECK}"
${soft_fail}
    timeout_in_minutes: 10
    env:
      EMUDHRA_SKIP_MITIGATION: "true"
    agents:
      image: "${image}"
STEP
    if [ -n "${uid}" ]; then
        echo "      imageUID: \"${uid}\""
    fi
}

cat <<EOL
steps:
EOL

# ml-linux-build:34 trusts the roots as-is (Rocky 8, ca-certificates 2024): a
# known gap until the image is rebuilt.
image_step linux_build "Linux build image (x86_64)" "docker.elastic.co/ml-dev/ml-linux-build:34" "" gap
image_step pytorch_dependency "PyTorch dependency image" "docker.elastic.co/ml-dev/ml-linux-dependency-build:pytorch_latest"
# The DRA steps run as non-root (imageUID 1000), so the mitigation can't apply:
# the image itself must be fixed (version 6 is Ubuntu 22.04).
image_step jdk17 "JDK 17 DRA image" "docker.elastic.co/ml-dev/ml_cpp_linux_x86_64_jdk17:6" 1000
image_step python "python:3.11-bookworm image" "python:3.11-bookworm"

# aarch64 builds run on an AlmaLinux 8 host, which post-checkout mitigates.
cat <<EOL
  - label: "CA trust check: Linux aarch64 build host"
    key: "check_ca_trust_linux_aarch64"
    command: "${CHECK}"
${SOFT_FAIL_INCONCLUSIVE}
    timeout_in_minutes: 10
${AARCH64_AGENTS}
EOL

# The aarch64 tests run in this image via "docker run", unmitigated, so this
# reports the image as-is.  A known, low-risk gap (it only runs unit tests), so
# soft-fail for now.  (ml-check-style is not checked: no TLS use.)
cat <<EOL
  - label: "CA trust check: Linux aarch64 native build image (test container)"
    key: "check_ca_trust_linux_aarch64_image"
    command:
      - "docker pull docker.elastic.co/ml-dev/ml-linux-aarch64-native-build:17"
      - "docker run --rm -v \"\$(pwd)/.buildkite/scripts:/ci:ro\" docker.elastic.co/ml-dev/ml-linux-aarch64-native-build:17 bash /ci/steps/check_ca_trust.sh"
    soft_fail: true
    timeout_in_minutes: 15
${AARCH64_AGENTS}
EOL

# macOS builds run natively on Orka VMs (same agent as build_macos.json.py).
cat <<EOL
  - label: "CA trust check: macOS aarch64 agent"
    key: "check_ca_trust_macos_aarch64"
    command: "${CHECK}"
${SOFT_FAIL_INCONCLUSIVE}
    timeout_in_minutes: 10
    agents:
      provider: "orka"
      image: "ml-macos-14-arm-003.orkasi"
EOL

# Windows builds run natively on GCP VMs (same agent as build_windows.json.py).
# Check only: Microsoft has already distrusted the CAs.  Exit 3 (only the JDK's
# cacerts or another tool's CA bundle trusts them) soft-fails too, as a known
# gap until the image is rebuilt.  "& script" alone reports any non-zero exit as 1, hence the explicit
# exit ($$ is Buildkite's escape for a literal $).
cat <<EOL
  - label: "CA trust check: Windows x86_64 agent"
    key: "check_ca_trust_windows_x86_64"
    command: '& .buildkite\scripts\steps\check_ca_trust.ps1; exit \$\$LASTEXITCODE'
    soft_fail:
      - exit_status: 2
      - exit_status: 3
    timeout_in_minutes: 10
    agents:
      provider: "gcp"
      machineType: "c2-standard-16"
      minCpuPlatform: "Intel Cascade Lake"
      image: "family/ml-cpp-5-windows-2022"
EOL
