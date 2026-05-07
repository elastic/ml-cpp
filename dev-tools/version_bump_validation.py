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
"""Rules for ml-cpp patch release version bump parameters (Buildkite / release-eng).

Used by dev-tools/validate_version_bump_params.sh and unit-tested under
dev-tools/unittest/.

Run tests from repo root (install dev-tools test deps first, see
``dev-tools/run_dev_tools_tests.sh``):

    python3 -m pip install -r dev-tools/test-requirements.txt
    ./dev-tools/run_dev_tools_tests.sh

Optional git integration (real ``git fetch`` + shell validator): set
``VERSION_BUMP_GIT_INTEGRATION=1`` and ``VERSION_BUMP_TEST_BRANCH=MAJOR.MINOR``.
See ``unittest/test_version_bump_validation.py`` module docstring.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Optional, Tuple

SEMVER_RE = re.compile(r"^([0-9]+)\.([0-9]+)\.([0-9]+)$")
BRANCH_RE = re.compile(r"^([0-9]+)\.([0-9]+)$")


def parse_semver(version: str) -> Optional[Tuple[int, int, int]]:
    m = SEMVER_RE.match(version.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def parse_release_branch(branch: str) -> Optional[Tuple[int, int]]:
    m = BRANCH_RE.match(branch.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def validate_version_bump_params(
    *,
    current_version: str,
    new_version: str,
    branch: str,
) -> None:
    """Validate patch bump parameters. Raises ValueError on failure.

    When current_version == new_version, the bump is a no-op and always valid.
    """
    new_t = parse_semver(new_version)
    if new_t is None:
        raise ValueError(
            f"NEW_VERSION must be MAJOR.MINOR.PATCH (digits only), got {new_version!r}"
        )
    new_major, new_minor, new_patch = new_t

    br = parse_release_branch(branch)
    if br is None:
        raise ValueError(
            f"BRANCH must be MAJOR.MINOR (e.g. 9.5), got {branch!r}"
        )
    br_major, br_minor = br
    if br_major != new_major or br_minor != new_minor:
        raise ValueError(
            f"BRANCH {branch!r} must match MAJOR.MINOR of NEW_VERSION "
            f"({new_major}.{new_minor}), got NEW_VERSION {new_version!r}"
        )

    cur_t = parse_semver(current_version)
    if cur_t is None:
        raise ValueError(
            "elasticsearchVersion on branch must be MAJOR.MINOR.PATCH, "
            f"got {current_version!r}"
        )
    cur_major, cur_minor, cur_patch = cur_t

    if current_version.strip() == new_version.strip():
        return

    if cur_major != new_major or cur_minor != new_minor:
        raise ValueError(
            "patch bump requires same MAJOR.MINOR as current "
            f"({cur_major}.{cur_minor} vs {new_major}.{new_minor})"
        )
    expected_patch = cur_patch + 1
    if new_patch != expected_patch:
        raise ValueError(
            "patch bump expects NEW_VERSION patch = current patch + 1 "
            f"({current_version} → {new_major}.{new_minor}.{expected_patch}), "
            f"got {new_version}"
        )


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        validate_version_bump_params(
            current_version=args.current,
            new_version=args.new,
            branch=args.branch,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


def _cmd_validate_and_report(args: argparse.Namespace) -> int:
    rc = _cmd_validate(args)
    if rc != 0:
        return rc
    cur = args.current.strip()
    new = args.new.strip()
    if cur == new:
        print(f"OK: branch already at {new} — bump step will no-op.")
    else:
        print(f"OK: patch increment {cur} → {new}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ml-cpp patch version bump parameter validation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser(
        "validate",
        help="check current/new/branch (same rules as Buildkite)",
    )
    p_val.add_argument("--current", required=True, help="elasticsearchVersion on branch")
    p_val.add_argument("--new", required=True, dest="new", help="NEW_VERSION")
    p_val.add_argument("--branch", required=True, help="BRANCH (MAJOR.MINOR)")
    p_val.set_defaults(func=_cmd_validate)

    p_rep = sub.add_parser(
        "validate-and-report",
        help="validate and print the same OK lines as validate_version_bump_params.sh",
    )
    p_rep.add_argument("--current", required=True)
    p_rep.add_argument("--new", required=True, dest="new")
    p_rep.add_argument("--branch", required=True)
    p_rep.set_defaults(func=_cmd_validate_and_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
