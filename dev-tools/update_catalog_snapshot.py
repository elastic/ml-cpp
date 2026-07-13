#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
"""Register a newly cut release branch in the ml-cpp snapshot build pipeline.

When a minor feature freeze cuts a new MAJOR.MINOR release branch, the daily
snapshot build pipeline defined in ``catalog-info.yaml`` must be told about it in
two places:

  * ``filter_condition`` -- otherwise scheduled/triggered builds for the branch
    are blocked.
  * ``schedules`` -- a new ``Daily <branch>`` cron entry so the branch gets a
    daily SNAPSHOT build.

The edit is deliberately a scoped, comment-preserving text transform (rather than
a YAML round-trip) so the rest of ``catalog-info.yaml`` -- comments, ordering and
formatting -- is left byte-for-byte identical. Only the snapshot pipeline block is
touched; the staging pipeline already matches any MAJOR.MINOR via a regex filter
and has no schedules, so it needs no change.

The operation is idempotent: re-running for a branch that is already registered
makes no change (important because the centralized version bump may re-trigger a
completed freeze).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Tuple

# Anchors the snapshot pipeline document. Unique within catalog-info.yaml.
_SNAPSHOT_ANCHOR = 'name: "ml-cpp-snapshot-builds"'
# Section boundaries are the human-authored "# Declare the ... pipeline" comments.
_SECTION_END_RE = re.compile(r"^# Declare ", re.MULTILINE)
_FILTER_RE = re.compile(r"^(?P<indent>[ \t]*)filter_condition:[ \t]*(?P<val>.*)$", re.MULTILINE)
_SCHEDULES_RE = re.compile(r"^(?P<indent>[ \t]*)schedules:[ \t]*$", re.MULTILINE)
_CRON_HOUR_RE = re.compile(r"cronline:[ \t]*30[ \t]+(\d{1,2})[ \t]")
_BRANCH_RE = re.compile(r"^[0-9]+\.[0-9]+$")


def _snapshot_bounds(text: str) -> Tuple[int, int]:
    start = text.find(_SNAPSHOT_ANCHOR)
    if start == -1:
        raise ValueError(f"snapshot pipeline anchor not found ({_SNAPSHOT_ANCHOR!r})")
    end_match = _SECTION_END_RE.search(text, start)
    end = end_match.start() if end_match else len(text)
    return start, end


def _update_filter_condition(section: str, branch: str) -> Tuple[str, bool]:
    m = _FILTER_RE.search(section)
    if m is None:
        raise ValueError("filter_condition not found in snapshot pipeline")
    val = m.group("val")
    token = f'build.branch == "{branch}"'
    if token in val:
        return section, False
    main_token = 'build.branch == "main"'
    if main_token in val:
        # Insert immediately after main to keep newest-first ordering.
        pos = val.index(main_token) + len(main_token)
        new_val = f"{val[:pos]} || {token}{val[pos:]}"
    else:
        new_val = f"{val} || {token}"
    new_line = f"{m.group('indent')}filter_condition: {new_val}"
    return section[: m.start()] + new_line + section[m.end() :], True


def _next_cron_hour(section: str) -> int:
    hours = [int(h) for h in _CRON_HOUR_RE.findall(section)]
    next_hour = (max(hours) + 1) if hours else 0
    if next_hour > 23:
        raise ValueError(
            "no free daily 'half past' snapshot slot: hours 0-23 are all taken; "
            "prune EOL branch schedules in catalog-info.yaml before adding more"
        )
    return next_hour


def _add_schedule(section: str, branch: str) -> Tuple[str, bool]:
    m = _SCHEDULES_RE.search(section)
    if m is None:
        raise ValueError("schedules block not found in snapshot pipeline")
    # Already scheduled? (accept single/double quoted or bare branch value)
    if re.search(rf"^[ \t]*branch:[ \t]*['\"]?{re.escape(branch)}['\"]?[ \t]*$", section, re.MULTILINE):
        return section, False

    schedules_indent = m.group("indent")
    key_indent = schedules_indent + "  "
    field_indent = key_indent + "  "
    cron_hour = _next_cron_hour(section)
    block = (
        f"{key_indent}Daily {branch}:\n"
        f"{field_indent}branch: '{branch}'\n"
        f"{field_indent}cronline: 30 {cron_hour:02d} * * *\n"
        f"{field_indent}message: Daily SNAPSHOT build for {branch}\n"
    )

    # Insert before the "Daily main:" entry so main stays last; otherwise append
    # after the last schedule entry (before the next same-or-lower indent key).
    main_entry = re.compile(rf"^{re.escape(key_indent)}Daily main:[ \t]*$", re.MULTILINE)
    main_m = main_entry.search(section, m.end())
    if main_m is not None:
        insert_at = main_m.start()
        return section[:insert_at] + block + section[insert_at:], True

    # Fallback: insert right after the schedules: line.
    insert_at = m.end() + 1  # past the trailing newline of the schedules: line
    return section[:insert_at] + block + section[insert_at:], True


def add_release_branch_to_snapshot(text: str, branch: str) -> Tuple[str, bool]:
    """Add ``branch`` to the snapshot pipeline filter and schedules. Idempotent.

    Returns (new_text, changed).
    """
    if not _BRANCH_RE.match(branch):
        raise ValueError(f"branch must be MAJOR.MINOR (e.g. 9.5), got {branch!r}")

    start, end = _snapshot_bounds(text)
    section = text[start:end]

    section, filter_changed = _update_filter_condition(section, branch)
    section, schedule_changed = _add_schedule(section, branch)

    changed = filter_changed or schedule_changed
    if not changed:
        return text, False
    return text[:start] + section + text[end:], True


def _cmd_update(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_file():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8")
    try:
        new_text, changed = add_release_branch_to_snapshot(text, args.branch)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not changed:
        print(f"OK: {path} already schedules snapshot builds for branch {args.branch}")
        return 0

    path.write_text(new_text, encoding="utf-8")
    print(f"Updated {path}: added branch {args.branch} to snapshot filter and schedules")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Register a release branch in the ml-cpp snapshot build pipeline"
    )
    parser.add_argument("--path", default="catalog-info.yaml", help="Path to catalog-info.yaml")
    parser.add_argument("--branch", required=True, help="Release branch (MAJOR.MINOR, e.g. 9.5)")
    args = parser.parse_args()
    return _cmd_update(args)


if __name__ == "__main__":
    sys.exit(main())
