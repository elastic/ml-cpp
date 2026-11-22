#!/usr/bin/env bash
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

# Run a command; if it is still running after the timeout, dump its thread
# stacks (so a hang can be diagnosed from the CI log), kill it and exit 124.
#
# Usage: run-with-watchdog.sh <timeout-seconds> <command> [args...]

timeout_s=$1
shift

# SIP removes DYLD_LIBRARY_PATH when bash starts on macOS; restore it for the command
if [ -n "${ML_WATCHDOG_DYLD_LIBRARY_PATH:-}" ]; then
    export DYLD_LIBRARY_PATH="${ML_WATCHDOG_DYLD_LIBRARY_PATH}"
fi

"$@" &
pid=$!

dump_diagnostics() {
    echo "=== WATCHDOG: pid ${pid} still running after ${timeout_s}s ==="
    date -u
    # Machine load, to tell a stalled host from a hung test
    uptime
    # Open files (a FIFO that was never connected shows up here)
    if command -v lsof >/dev/null 2>&1; then
        lsof -p "${pid}" 2>&1
    else
        ls -l /proc/"${pid}"/fd 2>&1
    fi
    # Only the command and its children, not the whole process table
    kids=$(pgrep -d, -P "${pid}")
    if [ "$(uname)" = Darwin ]; then
        ps -o pid,ppid,stat,etime,command -p "${pid}${kids:+,${kids}}"
        sample "${pid}" 3 2>&1
    else
        ps -o pid,ppid,stat,etime,wchan:24,args -p "${pid}${kids:+,${kids}}"
        for t in /proc/"${pid}"/task/*; do
            echo "--- thread ${t##*/} $(cat "${t}/comm" 2>/dev/null) wchan=$(cat "${t}/wchan" 2>/dev/null)"
        done
        if command -v gdb >/dev/null 2>&1; then
            gdb -p "${pid}" -batch -ex "thread apply all bt" 2>&1
        elif command -v eu-stack >/dev/null 2>&1; then
            eu-stack -p "${pid}" 2>&1
        fi
    fi
    echo "=== WATCHDOG: end of diagnostics ==="
}

# Print the PIDs of every descendant of $1, including processes that were put
# in their own process group or session.
descendants() {
    local child
    for child in $(pgrep -P "$1" 2>/dev/null); do
        echo "${child}"
        descendants "${child}"
    done
}

# SIGKILL $1 and everything below it. Freeze the tree first so nothing can fork
# or be reparented while the list is collected, then kill the whole list.
kill_tree() {
    local victims
    victims=$(descendants "$1")
    # shellcheck disable=SC2086
    kill -STOP "$1" ${victims} 2>/dev/null
    victims="${victims} $(descendants "$1")"
    # shellcheck disable=SC2086
    kill -9 ${victims} "$1" 2>/dev/null
}

# Diagnostics may hang (gdb, lsof), so they get a fixed budget, well inside the
# margin before CTest's own timeout.
diag_budget=${ML_WATCHDOG_DIAG_SECONDS:-15}

timed_out_flag=$(mktemp "${TMPDIR:-/tmp}/ml-watchdog.XXXXXX")
rm -f "${timed_out_flag}"

(
    sleep "${timeout_s}" &
    sleeper=$!
    trap 'kill "${sleeper}" 2>/dev/null; exit 0' TERM
    wait "${sleeper}"
    kill -0 "${pid}" 2>/dev/null || exit 0
    : > "${timed_out_flag}"
    dump_diagnostics &
    diag=$!
    waited=0
    while kill -0 "${diag}" 2>/dev/null && [ "${waited}" -lt "${diag_budget}" ]; do
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "${diag}" 2>/dev/null; then
        echo "=== WATCHDOG: diagnostics still running after ${diag_budget}s, abandoning them ==="
        kill_tree "${diag}"
    fi
    kill_tree "${pid}"
    exit 124
) &
watchdog=$!

wait "${pid}"
rc=$?

if [ ! -e "${timed_out_flag}" ]; then
    kill "${watchdog}" 2>/dev/null
fi
wait "${watchdog}" 2>/dev/null
if [ -e "${timed_out_flag}" ]; then
    rm -f "${timed_out_flag}"
    exit 124
fi
exit "${rc}"
