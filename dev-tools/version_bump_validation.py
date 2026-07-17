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

Patch and minor (feature freeze) workflows share parameter names from release-eng:
NEW_VERSION, BRANCH, WORKFLOW. For WORKFLOW=minor, NEW_VERSION is the version
expected on the new release branch (e.g. 9.5.0 on branch 9.5); main is bumped to
derive_main_new_version(NEW_VERSION) (e.g. 9.6.0).

BRANCH may be MAJOR.MINOR or a sandbox ref ``testing-MAJOR.MINOR`` (e.g. ``testing-9.5``).
Version rules strip the ``testing-`` prefix; git operations use the full ref name.

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
SANDBOX_BRANCH_PREFIX = "testing-"


def is_sandbox_release_branch(branch: str) -> bool:
    """True when BRANCH is a manual-test ref (testing-MAJOR.MINOR), not a production line."""
    return branch.startswith(SANDBOX_BRANCH_PREFIX)


def release_branch_identity(branch: str) -> str:
    """Return MAJOR.MINOR identity for version rules (strip leading testing- prefix)."""
    if is_sandbox_release_branch(branch):
        return branch[len(SANDBOX_BRANCH_PREFIX) :]
    return branch


def _reject_outer_whitespace(label: str, value: str) -> None:
    """Disallow leading/trailing ASCII whitespace or CR so values match shell expectations."""
    if value != value.strip(" \t\n\r\v\f"):
        raise ValueError(
            f"{label} must not have leading or trailing whitespace, got {value!r}"
        )


def parse_semver(version: str) -> Optional[Tuple[int, int, int]]:
    m = SEMVER_RE.match(version)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def parse_release_branch(branch: str) -> Optional[Tuple[int, int]]:
    identity = release_branch_identity(branch)
    m = BRANCH_RE.match(identity)
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
    _reject_outer_whitespace("NEW_VERSION", new_version)
    _reject_outer_whitespace("BRANCH", branch)
    _reject_outer_whitespace("current_version", current_version)

    new_t = parse_semver(new_version)
    if new_t is None:
        raise ValueError(
            f"NEW_VERSION must be MAJOR.MINOR.PATCH (digits only), got {new_version!r}"
        )
    new_major, new_minor, new_patch = new_t

    br = parse_release_branch(branch)
    if br is None:
        raise ValueError(
            f"BRANCH must be MAJOR.MINOR (e.g. 9.5) or "
            f"{SANDBOX_BRANCH_PREFIX}MAJOR.MINOR (e.g. testing-9.5), got {branch!r}"
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

    if current_version == new_version:
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


def derive_main_new_version(release_branch_version: str) -> str:
    """Return the main-branch version after minor feature freeze (minor + 1, patch 0)."""
    parsed = parse_semver(release_branch_version)
    if parsed is None:
        raise ValueError(
            f"release branch version must be MAJOR.MINOR.PATCH, got {release_branch_version!r}"
        )
    major, minor, patch = parsed
    if patch != 0:
        raise ValueError(
            "minor freeze expects NEW_VERSION patch 0 "
            f"(got {release_branch_version!r})"
        )
    return f"{major}.{minor + 1}.0"


def validate_minor_freeze_params(
    *,
    main_version: str,
    new_version: str,
    branch: str,
    release_branch_exists: bool,
    release_branch_version: str | None,
) -> str:
    """Validate minor freeze inputs. Returns MAIN_NEW_VERSION (derived).

    NEW_VERSION is the version expected on the new release branch (BRANCH).
    main must currently be at NEW_VERSION before the freeze bump.
    """
    _reject_outer_whitespace("NEW_VERSION", new_version)
    _reject_outer_whitespace("BRANCH", branch)
    _reject_outer_whitespace("main_version", main_version)

    new_t = parse_semver(new_version)
    if new_t is None:
        raise ValueError(
            f"NEW_VERSION must be MAJOR.MINOR.PATCH (digits only), got {new_version!r}"
        )
    new_major, new_minor, new_patch = new_t
    if new_patch != 0:
        raise ValueError(
            f"minor freeze NEW_VERSION must be X.Y.0 (patch 0), got {new_version!r}"
        )

    br = parse_release_branch(branch)
    if br is None:
        raise ValueError(
            f"BRANCH must be MAJOR.MINOR (e.g. 9.5) or "
            f"{SANDBOX_BRANCH_PREFIX}MAJOR.MINOR (e.g. testing-9.5), got {branch!r}"
        )
    br_major, br_minor = br
    if br_major != new_major or br_minor != new_minor:
        raise ValueError(
            f"BRANCH {branch!r} must match MAJOR.MINOR of NEW_VERSION "
            f"({new_major}.{new_minor}), got NEW_VERSION {new_version!r}"
        )

    main_t = parse_semver(main_version)
    if main_t is None:
        raise ValueError(
            "elasticsearchVersion on main must be MAJOR.MINOR.PATCH, "
            f"got {main_version!r}"
        )

    main_new_version = derive_main_new_version(new_version)

    if release_branch_exists:
        if release_branch_version is None:
            raise ValueError(
                f"release branch {branch!r} exists but version could not be read"
            )
        if release_branch_version != new_version:
            raise ValueError(
                f"release branch {branch!r} exists with version {release_branch_version!r}, "
                f"expected {new_version!r}"
            )

    # main may be at NEW_VERSION (freeze not yet applied) or, when the release branch
    # has already been cut, at MAIN_NEW_VERSION (freeze already completed). Accepting
    # the latter keeps re-runs of the centralized version bump idempotent instead of
    # failing after the freeze has already succeeded (see the centralized version-bump
    # pipeline re-triggering completed bumps). If main has advanced without the branch
    # being cut, that is a genuinely inconsistent state and is still rejected.
    if main_version == new_version:
        pass
    elif main_version == main_new_version and release_branch_exists:
        pass
    else:
        raise ValueError(
            "minor freeze requires main elasticsearchVersion to be NEW_VERSION before branching, "
            "or MAIN_NEW_VERSION only if the release branch already exists "
            f"(main={main_version!r}, NEW_VERSION={new_version!r}, MAIN_NEW_VERSION={main_new_version!r}, "
            f"release_branch_exists={release_branch_exists})"
        )

    return main_new_version


def validate_main_minor_bump(
    *,
    current_version: str,
    main_new_version: str,
    release_branch_version: str,
) -> None:
    """Validate bumping main from release-branch version to MAIN_NEW_VERSION."""
    _reject_outer_whitespace("current_version", current_version)
    _reject_outer_whitespace("main_new_version", main_new_version)
    _reject_outer_whitespace("release_branch_version", release_branch_version)

    if current_version == main_new_version:
        return

    if current_version != release_branch_version:
        raise ValueError(
            "main bump expects current main version to equal NEW_VERSION "
            f"({release_branch_version!r}), got {current_version!r}"
        )

    expected = derive_main_new_version(release_branch_version)
    if main_new_version != expected:
        raise ValueError(
            f"MAIN_NEW_VERSION must be {expected!r} for NEW_VERSION "
            f"{release_branch_version!r}, got {main_new_version!r}"
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
    cur = args.current
    new = args.new
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
    p_val.add_argument("--branch", required=True, help="BRANCH (MAJOR.MINOR or testing-MAJOR.MINOR)")
    p_val.set_defaults(func=_cmd_validate)

    p_rep = sub.add_parser(
        "validate-and-report",
        help="validate and print the same OK lines as validate_version_bump_params.sh",
    )
    p_rep.add_argument("--current", required=True)
    p_rep.add_argument("--new", required=True, dest="new")
    p_rep.add_argument("--branch", required=True)
    p_rep.set_defaults(func=_cmd_validate_and_report)

    p_minor = sub.add_parser(
        "validate-minor-freeze",
        help="check main/new/branch for WORKFLOW=minor",
    )
    p_minor.add_argument("--main-version", required=True)
    p_minor.add_argument("--new", required=True, dest="new")
    p_minor.add_argument("--branch", required=True)
    p_minor.add_argument(
        "--release-branch-exists",
        action="store_true",
        help="origin/BRANCH already exists",
    )
    p_minor.add_argument(
        "--release-branch-version",
        default="",
        help="elasticsearchVersion on origin/BRANCH when it exists",
    )

    def _cmd_validate_minor(args_ns: argparse.Namespace) -> int:
        try:
            rb_ver = args_ns.release_branch_version or None
            main_new = validate_minor_freeze_params(
                main_version=args_ns.main_version,
                new_version=args_ns.new,
                branch=args_ns.branch,
                release_branch_exists=args_ns.release_branch_exists,
                release_branch_version=rb_ver,
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print(f"OK: minor freeze NEW_VERSION={args_ns.new} on branch {args_ns.branch}")
        if args_ns.main_version == main_new:
            print(
                f"OK: main already bumped to {main_new} and branch {args_ns.branch} "
                "exists — minor freeze already completed (idempotent no-op)."
            )
        if is_sandbox_release_branch(args_ns.branch):
            identity = release_branch_identity(args_ns.branch)
            print(
                f"OK: sandbox branch (version identity {identity!r}); "
                "main bump and DRA wait are skipped in CI"
            )
        print(f"OK: main bump target MAIN_NEW_VERSION={main_new}")
        return 0

    p_minor.set_defaults(func=_cmd_validate_minor)

    p_main = sub.add_parser(
        "validate-main-minor-bump",
        help="check main bump during minor freeze",
    )
    p_main.add_argument("--current", required=True)
    p_main.add_argument("--main-new-version", required=True)
    p_main.add_argument("--release-branch-version", required=True)

    def _cmd_validate_main_bump(args_ns: argparse.Namespace) -> int:
        try:
            validate_main_minor_bump(
                current_version=args_ns.current,
                main_new_version=args_ns.main_new_version,
                release_branch_version=args_ns.release_branch_version,
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        if args_ns.current == args_ns.main_new_version:
            print(f"OK: main already at {args_ns.main_new_version} — bump step will no-op.")
        else:
            print(
                f"OK: main minor bump {args_ns.current} → {args_ns.main_new_version}"
            )
        return 0

    p_main.set_defaults(func=_cmd_validate_main_bump)

    p_derive = sub.add_parser(
        "derive-main-new-version",
        help="print MAIN_NEW_VERSION for NEW_VERSION (minor freeze)",
    )
    p_derive.add_argument("--new", required=True, dest="new")

    def _cmd_derive_main(args_ns: argparse.Namespace) -> int:
        try:
            print(derive_main_new_version(args_ns.new))
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        return 0

    p_derive.set_defaults(func=_cmd_derive_main)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
