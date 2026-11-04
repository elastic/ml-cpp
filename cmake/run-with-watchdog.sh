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

"$@" &
pid=$!

dump_diagnostics() {
    echo "=== WATCHDOG: pid ${pid} still running after ${timeout_s}s ==="
    date -u
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

(
    sleep "${timeout_s}" &
    sleeper=$!
    trap 'kill "${sleeper}" 2>/dev/null; exit 0' TERM
    wait "${sleeper}"
    kill -0 "${pid}" 2>/dev/null || exit 0
    dump_diagnostics
    pkill -9 -P "${pid}" 2>/dev/null
    kill -9 "${pid}" 2>/dev/null
    exit 124
) &
watchdog=$!

wait "${pid}"
rc=$?

kill "${watchdog}" 2>/dev/null
wait "${watchdog}" 2>/dev/null
[ $? -eq 124 ] && exit 124
exit "${rc}"
