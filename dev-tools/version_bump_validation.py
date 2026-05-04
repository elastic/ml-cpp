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
"""Rules for ml-cpp release version bump parameters (Buildkite / release-eng).

Used by dev-tools/validate_version_bump_params.sh and unit-tested under
dev-tools/unittest/.

Run tests from ``dev-tools/``:

    ./run_dev_tools_tests.sh

Or (after ``pip install -r dev-tools/test-requirements.txt``):

    cd dev-tools && python3 -m pytest -c pytest.ini
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


def validate_workflow_name(workflow: str) -> None:
    if workflow not in ("patch", "minor"):
        raise ValueError(
            f"WORKFLOW must be 'patch' or 'minor', got {workflow!r}"
        )


def validate_version_bump_params(
    *,
    current_version: str,
    new_version: str,
    branch: str,
    workflow: str,
) -> None:
    """Validate release bump parameters. Raises ValueError on failure.

    When current_version == new_version, the bump is a no-op and always valid.
    """
    validate_workflow_name(workflow)

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

    if workflow == "patch":
        if cur_major != new_major or cur_minor != new_minor:
            raise ValueError(
                "patch bump requires same MAJOR.MINOR as current "
                f"({cur_major}.{cur_minor} vs {new_major}.{new_minor})"
            )
        expected_patch = cur_patch + 1
        if new_patch != expected_patch:
            raise ValueError(
                "patch workflow expects NEW_VERSION patch = current patch + 1 "
                f"({current_version} → {new_major}.{new_minor}.{expected_patch}), "
                f"got {new_version}"
            )
        return

    # minor
    if new_patch != 0:
        raise ValueError(
            f"minor workflow expects NEW_VERSION with PATCH=0, got {new_version!r}"
        )
    if cur_major != new_major:
        raise ValueError(
            f"minor bump must keep the same MAJOR ({cur_major} vs {new_major})"
        )
    expected_minor = cur_minor + 1
    if new_minor != expected_minor:
        raise ValueError(
            "minor workflow expects MINOR = current minor + 1 "
            f"({cur_minor} → {expected_minor}), got {new_minor}"
        )


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        validate_version_bump_params(
            current_version=args.current,
            new_version=args.new,
            branch=args.branch,
            workflow=args.workflow,
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
    elif args.workflow == "patch":
        print(f"OK: patch increment {cur} → {new}")
    else:
        print(f"OK: minor increment {cur} → {new}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ml-cpp version bump parameter validation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser(
        "validate",
        help="check current/new/branch/workflow (same rules as Buildkite)",
    )
    p_val.add_argument("--current", required=True, help="elasticsearchVersion on branch")
    p_val.add_argument("--new", required=True, dest="new", help="NEW_VERSION")
    p_val.add_argument("--branch", required=True, help="BRANCH (MAJOR.MINOR)")
    p_val.add_argument(
        "--workflow",
        required=True,
        choices=("patch", "minor"),
        help="WORKFLOW",
    )
    p_val.set_defaults(func=_cmd_validate)

    p_rep = sub.add_parser(
        "validate-and-report",
        help="validate and print the same OK lines as validate_version_bump_params.sh",
    )
    p_rep.add_argument("--current", required=True)
    p_rep.add_argument("--new", required=True, dest="new")
    p_rep.add_argument("--branch", required=True)
    p_rep.add_argument("--workflow", required=True, choices=("patch", "minor"))
    p_rep.set_defaults(func=_cmd_validate_and_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
