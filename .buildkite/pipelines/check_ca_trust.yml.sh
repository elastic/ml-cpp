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
# Runs check_ca_trust.sh in each CI Docker image, on the Linux build hosts and
# on the macOS agent, and check_ca_trust.ps1 on the Windows agent, to check
# whether they trust the compromised eMudhra CAs (InfoSec incident "Intrepid
# Falcon").  The host steps run after the post-checkout mitigation, so PASS
# means it took effect.  Exit 1 (a store still trusts them) fails the build;
# exit 2 (inconclusive, e.g. the test site is unreachable) only warns.  The
# CentOS 7 images trust them as shipped and have no fixed package: those steps
# are known gaps (soft-fail), and the mitigation covers the steps that run in
# them.

CHECK=".buildkite/scripts/steps/check_ca_trust.sh"

SOFT_FAIL_INCONCLUSIVE='    soft_fail:
      - exit_status: 2'

AARCH64_AGENTS='    agents:
      provider: "aws"
      instanceType: "m6g.2xlarge"
      imagePrefix: "ci-amazonlinux-2-aarch64"
      diskSizeGb: "100"
      diskName: "/dev/xvda"'

# Emits a step that runs the check inside the given Docker image, without the
# post-checkout mitigation, so it reports the image as-is.  A fifth argument
# "gap" soft-fails the step: a known gap.  A sixth argument "mitigated" runs the
# step with the mitigation instead, as the build steps that use the image do.
image_step() {
    local key="$1" label="$2" image="$3" uid="${4:-}" gap="${5:-}" mode="${6:-}"
    local soft_fail="${SOFT_FAIL_INCONCLUSIVE}"
    [ "${gap}" = gap ] && soft_fail='    soft_fail: true'
    cat <<STEP
  - label: "CA trust check: ${label}"
    key: "check_ca_trust_${key}"
    command: "${CHECK}"
${soft_fail}
    timeout_in_minutes: 10
STEP
    [ "${mode}" = mitigated ] || printf '    env:\n      EMUDHRA_SKIP_MITIGATION: "true"\n'
    echo "    agents:"
    echo "      image: \"${image}\""
    if [ -n "${uid}" ]; then
        echo "      imageUID: \"${uid}\""
    fi
}

cat <<EOL
steps:
EOL

# The x86_64 build and the aarch64 cross-compile steps run in these images; they
# are CentOS 7, which has no fixed ca-certificates package.
image_step linux_build "Linux build image (x86_64)" "docker.elastic.co/ml-dev/ml-linux-build:32a" "" gap
image_step linux_aarch64_cross "Linux aarch64 cross-build image" "docker.elastic.co/ml-dev/ml-linux-aarch64-cross-build:15a" "" gap
# The same two images with the mitigation, as the build steps use them: the pods
# run as root, so post-checkout can blacklist the roots in the image's CA store.
image_step linux_build_mitigated "Linux build image (x86_64), mitigated" "docker.elastic.co/ml-dev/ml-linux-build:32a" "" "" mitigated
image_step linux_aarch64_cross_mitigated "Linux aarch64 cross-build image, mitigated" "docker.elastic.co/ml-dev/ml-linux-aarch64-cross-build:15a" "" "" mitigated
# The DRA steps run as non-root (imageUID 1000), so the mitigation can't apply:
# the image itself must be fixed (version 6 is Ubuntu 22.04).
image_step jdk17 "JDK 17 DRA image" "docker.elastic.co/ml-dev/ml_cpp_linux_x86_64_jdk17:6" 1000
image_step python "python:3.11-bookworm image" "python:3.11-bookworm"

# The native aarch64 builds and the Java integration tests run on Amazon Linux 2
# hosts, which post-checkout mitigates.
cat <<EOL
  - label: "CA trust check: Linux aarch64 build host"
    key: "check_ca_trust_linux_aarch64"
    command: "${CHECK}"
${SOFT_FAIL_INCONCLUSIVE}
    timeout_in_minutes: 10
${AARCH64_AGENTS}
  - label: "CA trust check: Linux x86_64 test host"
    key: "check_ca_trust_linux_x86_64_host"
    command: "${CHECK}"
${SOFT_FAIL_INCONCLUSIVE}
    timeout_in_minutes: 10
    agents:
      provider: "aws"
      instanceType: "m6i.2xlarge"
      imagePrefix: "ci-amazonlinux-2"
      diskSizeGb: "100"
      diskName: "/dev/xvda"
EOL

# The aarch64 tests run in this image via "docker run", unmitigated, so this
# reports the image as-is.  A known gap: CentOS 7.  The second step runs the
# distrust script first, as docker_entrypoint.sh now does when the container is
# built, to check that it takes effect there.
# (ml-check-style is not checked: no TLS use.)
cat <<EOL
  - label: "CA trust check: Linux aarch64 native build image (test container)"
    key: "check_ca_trust_linux_aarch64_image"
    command:
      - "docker pull docker.elastic.co/ml-dev/ml-linux-aarch64-native-build:15a"
      - "docker run --rm -v \"\$(pwd)/.buildkite/scripts:/ci:ro\" docker.elastic.co/ml-dev/ml-linux-aarch64-native-build:15a bash /ci/steps/check_ca_trust.sh"
    soft_fail: true
    timeout_in_minutes: 15
${AARCH64_AGENTS}
  - label: "CA trust check: Linux aarch64 native build image (test container), mitigated"
    key: "check_ca_trust_linux_aarch64_image_mitigated"
    command:
      - "docker pull docker.elastic.co/ml-dev/ml-linux-aarch64-native-build:15a"
      - "docker run --rm -v \"\$(pwd)/.buildkite/scripts:/ci:ro\" docker.elastic.co/ml-dev/ml-linux-aarch64-native-build:15a bash -c 'bash /ci/common/distrust_emudhra_cas.sh; bash /ci/steps/check_ca_trust.sh'"
${SOFT_FAIL_INCONCLUSIVE}
    timeout_in_minutes: 15
${AARCH64_AGENTS}
EOL

# macOS builds run natively on Orka VMs (same agent as build_macos.json.py).  The
# macOS 12 image (-004) has the roots distrusted in the image, so this reports it
# as-is.
cat <<EOL
  - label: "CA trust check: macOS aarch64 agent"
    key: "check_ca_trust_macos_aarch64"
    command: "${CHECK}"
${SOFT_FAIL_INCONCLUSIVE}
    timeout_in_minutes: 10
    env:
      EMUDHRA_SKIP_MITIGATION: "true"
    agents:
      provider: "orka"
      image: "ml-macos-12-arm-004.orkasi"
EOL

# Windows builds run natively on GCP VMs (same agent as build_windows.json.py).
# Check only: Microsoft has already distrusted the CAs.  Exit 3 (only the JDK's
# cacerts or another tool's CA bundle trusts them) fails the build: the image
# (ml-cpp-3-windows-2016) has them distrusted.  "& script" alone reports any
# non-zero exit as 1, hence the explicit exit ($$ is Buildkite's escape for a
# literal $).
cat <<EOL
  - label: "CA trust check: Windows x86_64 agent"
    key: "check_ca_trust_windows_x86_64"
    command: '& .buildkite\scripts\steps\check_ca_trust.ps1; exit \$\$LASTEXITCODE'
${SOFT_FAIL_INCONCLUSIVE}
    timeout_in_minutes: 10
    agents:
      provider: "gcp"
      machineType: "c2-standard-16"
      minCpuPlatform: "Intel Cascade Lake"
      image: "family/ml-cpp-3-windows-2016"
EOL
