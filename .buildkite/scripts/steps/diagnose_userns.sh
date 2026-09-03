#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Reports whether this agent can create the user namespace Sandbox2 needs, and
# if not, which stage the kernel denies.
#
# Sandbox2 requires all of: unshare(CLONE_NEWUSER), a uid/gid mapping,
# unshare(CLONE_NEWNS), and a fresh mount of procfs. Any one of those can be
# denied independently, and the failures look identical from outside - which is
# why this reports them stage by stage rather than as one yes/no.
#
# Needs no build artifacts, so it runs with no depends_on and answers in about a
# minute instead of waiting behind a 45 minute compile. Never fails the step:
# this is instrumentation, not a gate.

set -uo pipefail

# util-linux --map-root-user performs the same uid_map/setgroups/gid_map writes
# as the C++ probe in lib/sandbox/unittest/CSandboxedProcessSpawnerTest_Linux.cc,
# so a stage that fails here fails there too.
PROBE_STAGES=(
    "unshare(CLONE_NEWUSER)|unshare --user true"
    "uid/gid mapping|unshare --user --map-root-user true"
    "unshare(CLONE_NEWNS)|unshare --user --map-root-user --mount true"
    "mount(proc)|unshare --user --map-root-user --mount sh -c 'mount --make-rprivate / && mount -t proc proc /proc'"
)

report_host_facts() {
    echo "uname:      $(uname -srm)"
    echo "os-release: $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}")"
    echo "id:         $(id)"

    local sysctl_path
    for sysctl_path in /proc/sys/user/max_user_namespaces \
                       /proc/sys/user/max_mnt_namespaces \
                       /proc/sys/kernel/unprivileged_userns_clone; do
        if [ -r "$sysctl_path" ]; then
            echo "${sysctl_path}: $(cat "$sysctl_path")"
        else
            echo "${sysctl_path}: <absent>"
        fi
    done

    if [ -r /sys/fs/selinux/enforce ]; then
        echo "selinux.enforce: $(cat /sys/fs/selinux/enforce)"
    else
        echo "selinux.enforce: <absent>"
    fi
    echo "apparmor: $([ -d /sys/kernel/security/apparmor ] && echo present || echo absent)"

    # The kernel refuses mount(proc) inside a new user namespace when any mount
    # covers part of /proc, even though CLONE_NEWUSER itself succeeded. On EC2
    # AlmaLinux that is usually /proc/sys/fs/binfmt_misc.
    local covering
    covering=$(awk '$5 ~ /^\/proc\// { print $5 }' /proc/self/mountinfo 2>/dev/null | paste -sd, -)
    echo "mounts covering /proc: ${covering:-none}"
}

run_probe_stages() {
    local runner_label="$1"
    shift

    local stage
    for stage in "${PROBE_STAGES[@]}"; do
        local name="${stage%%|*}"
        local command="${stage#*|}"
        if "$@" sh -c "$command" >/dev/null 2>&1; then
            echo "  ${runner_label} ${name}: OK"
        else
            echo "  ${runner_label} ${name}: DENIED (exit $?)"
            echo "  ${runner_label} first denied stage is ${name}; later stages not attempted"
            return 0
        fi
    done
    echo "  ${runner_label} all stages OK -> this runner supports ML_SANDBOX2_REQUIRE=enforced"
}

echo "--- Sandbox2 user-namespace diagnosis: host facts"
report_host_facts

if ! command -v unshare >/dev/null 2>&1; then
    echo "warning: util-linux unshare not on PATH; cannot stage the probe on the host"
else
    echo "--- Sandbox2 user-namespace diagnosis: host probe"
    run_probe_stages "host" env
fi

# The same kernel seen through the container the tests actually run in. A
# difference between the two rows is a daemon/runtime restriction rather than a
# kernel one.
BASE_IMAGE="${SANDBOX2_DIAGNOSE_IMAGE:-docker.elastic.co/ml-dev/ml-linux-aarch64-native-build:17}"
if ! command -v docker >/dev/null 2>&1; then
    echo "warning: docker not on PATH; skipping container probe"
elif ! docker pull "$BASE_IMAGE" >/dev/null 2>&1; then
    echo "warning: could not pull ${BASE_IMAGE}; skipping container probe"
else
    echo "--- Sandbox2 user-namespace diagnosis: container probe (${BASE_IMAGE})"
    run_probe_stages "docker" docker run --rm "$BASE_IMAGE"
    echo "--- Sandbox2 user-namespace diagnosis: container probe (--privileged)"
    run_probe_stages "docker-privileged" docker run --rm --privileged "$BASE_IMAGE"
fi

echo "--- Sandbox2 user-namespace diagnosis: done"
exit 0
