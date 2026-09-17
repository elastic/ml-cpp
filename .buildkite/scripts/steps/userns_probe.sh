#!/bin/bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Reports the namespace facts and staged probe results for whichever runner
# executes it - the host, or a container. diagnose_userns.sh feeds this script on
# stdin to each runner so host and container results are directly comparable.
#
# Stages mirror probeUserNamespaces() in
# lib/sandbox/unittest/CSandboxedProcessSpawnerTest_Linux.cc one for one, and
# util-linux --map-root-user performs the same uid_map/setgroups/gid_map writes,
# so a stage that fails here fails there. Each stage's stderr is reported: an
# exit status alone cannot tell a kernel denial from a missing binary.
#
# Always exits 0. This is instrumentation, not a gate.

set -uo pipefail

LABEL="${1:-runner}"

say() {
    echo "[${LABEL}] $*"
}

say "uname:      $(uname -srm)"
say "os-release: $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}")"
say "id:         $(id)"

for sysctl_path in /proc/sys/user/max_user_namespaces \
                   /proc/sys/user/max_mnt_namespaces \
                   /proc/sys/kernel/unprivileged_userns_clone; do
    if [ -r "$sysctl_path" ]; then
        say "${sysctl_path}: $(cat "$sysctl_path")"
    else
        say "${sysctl_path}: <absent>"
    fi
done

if [ -r /sys/fs/selinux/enforce ]; then
    say "selinux.enforce: $(cat /sys/fs/selinux/enforce)"
else
    say "selinux.enforce: <absent>"
fi

# The kernel refuses a fresh procfs inside a non-init user namespace while any
# mount covers part of /proc. Reported per runner because a container has its own
# /proc instance and its own masked paths, so this differs from the host.
covering=$(awk '$5 ~ /^\/proc\// { print $5 }' /proc/self/mountinfo 2>/dev/null | paste -sd, -)
say "mounts covering /proc: ${covering:-none}"
say "unshare binary: $(command -v unshare || echo '<absent>')"

if ! command -v unshare >/dev/null 2>&1; then
    say "cannot stage the probe: util-linux unshare is absent"
    exit 0
fi

# Each stage is a superset of the one before it, so the first failure is the
# blocker and later stages carry no extra information. Stage names contain '|',
# hence '@@' as the field separator.
# CLONE_NEWPID is required for the final stage, not optional: the kernel refuses
# a procfs instance for a PID namespace that already has one mounted, so
# mount(proc) in a user namespace returns EPERM unless the PID namespace is new
# too. --fork is what puts the mounting process inside it, since CLONE_NEWPID
# only affects children.
STAGES=(
    "unshare(CLONE_NEWUSER)@@unshare --user true"
    "uid/gid mapping@@unshare --user --map-root-user true"
    "unshare(CLONE_NEWNS)@@unshare --user --map-root-user --mount true"
    "unshare(CLONE_NEWPID)@@unshare --user --map-root-user --mount --pid --fork true"
    "mount(/, MS_REC|MS_PRIVATE)@@unshare --user --map-root-user --mount --pid --fork sh -c 'mount --make-rprivate /'"
    "mount(proc, /proc, proc)@@unshare --user --map-root-user --mount --pid --fork sh -c 'mount --make-rprivate / && mount -t proc proc /proc'"
)

for stage in "${STAGES[@]}"; do
    name="${stage%%@@*}"
    command="${stage#*@@}"
    if stderr=$(eval "$command" 2>&1 >/dev/null); then
        say "${name}: OK"
    else
        status=$?
        say "${name}: DENIED (exit ${status}) ${stderr:-<no stderr>}"
        say "first blocked stage: ${name}"
        exit 0
    fi
done

say "all stages OK -> this runner supports ML_SANDBOX2_REQUIRE=enforced"

# Contrast case, recorded so the CLONE_NEWPID requirement above stays visible
# rather than looking like an arbitrary extra flag. A failure here is expected
# and is not a problem with the runner.
if unshare --user --map-root-user --mount sh -c 'mount --make-rprivate / && mount -t proc proc /proc' >/dev/null 2>&1; then
    say "note: mount(proc) also succeeds without CLONE_NEWPID on this kernel"
else
    say "note: mount(proc) fails without CLONE_NEWPID, as expected"
fi

exit 0
