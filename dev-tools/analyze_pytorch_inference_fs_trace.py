#!/usr/bin/env python3
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
"""Summarize a trace_pytorch_inference_fs.bt log into a Landlock ruleset review.

Reads the raw trace on stdin or from a file argument and prints, per traced
process:

  * every filesystem path accessed AFTER the Landlock ruleset took effect
    (the LANDLOCK_RESTRICT_SELF marker), deduplicated, with whether each
    access succeeded or was denied - this is the exact set the ruleset must
    account for;
  * every DENIED access after that point (ret == -EACCES / -EPERM), which on
    a correctly derived ruleset must be empty: a denial here is either a path
    that needs granting or, worse, one whose denial is silently tolerated
    (e.g. /proc/cpuinfo, which makes oneMKL pick a slower kernel without
    failing).

When a trace has no LANDLOCK_RESTRICT_SELF marker (an unconfined baseline
run, taken with xpack.ml.trained_models.sandbox_enabled=false to see the full
access set), everything from the log-pipe open onward is reported instead, so
the confined and baseline runs can be diffed.

Usage:
  analyze_pytorch_inference_fs_trace.py [trace.log]
  bpftrace trace_pytorch_inference_fs.bt | analyze_pytorch_inference_fs_trace.py
"""
import re
import sys
from collections import OrderedDict

EACCES, EPERM = 13, 1

ACCESS_RE = re.compile(
    r'^\d+ (?P<pid>\d+) (?P<tid>\d+) (?P<op>\w+) flags=0x(?P<flags>[0-9a-f]+) '
    r'ret=(?P<ret>-?\d+) (?P<path>.+)$')
MARKER_RE = re.compile(r'^\d+ (?P<pid>\d+) \d+ LANDLOCK_RESTRICT_SELF ret=(?P<ret>-?\d+)$')
EXEC_RE = re.compile(r'^\d+ (?P<pid>\d+) EXEC (?P<path>.+)$')


def normalize(path):
    """Collapse the parts of a path that vary run to run, so distinct
    accesses to the "same" logical location deduplicate."""
    path = re.sub(r'/ml-child-ipc/[^/]+', '/ml-child-ipc/<deployment-id>', path)
    path = re.sub(r'/pytorch_inference_[^/ ]+', '/pytorch_inference_<pipe>', path)
    path = re.sub(r'/proc/\d+/', '/proc/<pid>/', path)
    path = re.sub(r'/proc/self/task/\d+', '/proc/self/task/<tid>', path)
    return path


def main():
    lines = (open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin).read().splitlines()

    # Per pid: the marker line index (or None), and the ordered access list.
    restrict_at = {}
    accesses = OrderedDict()  # pid -> list of (idx, op, flags, ret, path)
    for idx, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if m:
            restrict_at.setdefault(int(m.group('pid')), idx)
            continue
        m = ACCESS_RE.match(line)
        if m:
            pid = int(m.group('pid'))
            accesses.setdefault(pid, []).append(
                (idx, m.group('op'), m.group('flags'), int(m.group('ret')), m.group('path')))

    if not accesses:
        print("No pytorch_inference filesystem accesses found in the trace.", file=sys.stderr)
        return 1

    for pid, entries in accesses.items():
        marker = restrict_at.get(pid)
        if marker is not None:
            scope = [e for e in entries if e[0] > marker]
            header = f"pid {pid}: {len(scope)} accesses AFTER Landlock took effect"
        else:
            # No ruleset applied - baseline run. Report from the log-pipe open,
            # which is roughly where the confined run would apply the ruleset.
            first_log = next((e[0] for e in entries
                              if 'log' in e[4] and 'ml-child-ipc' in e[4] or '_log_' in e[4]), None)
            scope = [e for e in entries if first_log is None or e[0] >= first_log]
            header = (f"pid {pid}: no Landlock marker (baseline run); "
                      f"{len(scope)} accesses from the log-pipe open onward")

        print(f"\n=== {header} ===")
        seen = OrderedDict()
        denied = OrderedDict()
        for _, op, flags, ret, path in scope:
            key = (op, normalize(path))
            verdict = 'ok' if ret >= 0 else f'DENIED(errno={-ret})'
            seen.setdefault(key, verdict)
            if ret in (-EACCES, -EPERM):
                denied.setdefault(key, verdict)

        for (op, path), verdict in sorted(seen.items(), key=lambda kv: kv[0][1]):
            print(f"  {verdict:16} {op:8} {path}")

        print(f"  --- denials after the ruleset: {len(denied)} "
              f"(must be 0 on a correct ruleset) ---")
        for (op, path), verdict in sorted(denied.items(), key=lambda kv: kv[0][1]):
            print(f"  {verdict:16} {op:8} {path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
