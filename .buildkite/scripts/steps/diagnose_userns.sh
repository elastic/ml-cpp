#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Reports which runner on this agent, if any, can create the user namespace
# Sandbox2 needs - and for the ones that cannot, the exact stage the kernel
# denies.
#
# Sandbox2 requires all of: unshare(CLONE_NEWUSER), a uid/gid mapping,
# unshare(CLONE_NEWNS), a private "/", and a fresh mount of procfs. Any one can
# be denied independently and the failures are indistinguishable from outside,
# which is why userns_probe.sh reports them stage by stage in each runner.
#
# Runs the same probe on the host and under several container configurations,
# because the blocker differs between them: Docker's default seccomp profile
# denies unshare(CLONE_NEWUSER) outright, while a mount covering /proc blocks
# only the final stage. The point is to find one runner where every stage passes,
# which is where ML_SANDBOX2_REQUIRE=enforced can then be set.
#
# Needs no build artifacts, so it runs with no depends_on and answers in about a
# minute instead of waiting behind a 45 minute compile. Never fails the step:
# this is instrumentation, not a gate.

set -uo pipefail

MY_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="${MY_DIR}/userns_probe.sh"
BASE_IMAGE="${SANDBOX2_DIAGNOSE_IMAGE:-docker.elastic.co/ml-dev/ml-linux-aarch64-native-build:17}"

echo "--- Sandbox2 userns diagnosis: host"
bash "$PROBE" host

if ! command -v docker >/dev/null 2>&1; then
    echo "warning: docker not on PATH; skipping container probes"
    echo "--- Sandbox2 userns diagnosis: done"
    exit 0
fi

if ! docker pull "$BASE_IMAGE" >/dev/null 2>&1; then
    echo "warning: could not pull ${BASE_IMAGE}; skipping container probes"
    echo "--- Sandbox2 userns diagnosis: done"
    exit 0
fi

# label@@docker flags. Ordered least to most privileged: the cheapest
# configuration that clears every stage is the one to adopt, since each
# escalation widens what a compromised test process could reach.
RUNNERS=(
    "docker@@"
    "docker+seccomp-unconfined@@--security-opt seccomp=unconfined"
    "docker+seccomp+systempaths@@--security-opt seccomp=unconfined --security-opt systempaths=unconfined"
    "docker-privileged@@--privileged"
    "docker-privileged+systempaths@@--privileged --security-opt systempaths=unconfined"
)

for runner in "${RUNNERS[@]}"; do
    label="${runner%%@@*}"
    flags="${runner#*@@}"
    echo "--- Sandbox2 userns diagnosis: ${label}"
    # shellcheck disable=SC2086 - flags must word-split into separate arguments.
    docker run --rm -i $flags "$BASE_IMAGE" bash -s "$label" < "$PROBE" ||
        echo "[${label}] container probe could not run (exit $?)"
done

echo "--- Sandbox2 userns diagnosis: done"
exit 0
