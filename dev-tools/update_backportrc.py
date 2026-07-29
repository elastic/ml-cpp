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
"""Update .backportrc.json for minor release feature freeze."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_MAIN_NEW_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")


def update_backportrc_for_minor_freeze(
    data: dict[str, Any],
    *,
    new_release_branch: str,
    main_new_version: str,
) -> bool:
    """Apply minor-freeze updates in place. Returns True if anything changed.

    Raises ValueError if main_new_version is not MAJOR.MINOR.PATCH: a value like
    "9.6" would produce the key ^v9.6$, which never matches a v9.6.0 label, so the
    main override would silently never fire.
    """
    if not _MAIN_NEW_VERSION_RE.fullmatch(main_new_version):
        raise ValueError(
            "main_new_version must be MAJOR.MINOR.PATCH (e.g. 9.6.0), "
            f"got {main_new_version!r}"
        )

    changed = False

    choices: list[str] = list(data.get("targetBranchChoices", []))
    if new_release_branch not in choices:
        if "main" in choices:
            insert_at = choices.index("main") + 1
        else:
            insert_at = 0
        choices.insert(insert_at, new_release_branch)
        data["targetBranchChoices"] = choices
        changed = True

    old_mapping: dict[str, str] = dict(data.get("branchLabelMapping", {}))
    new_main_key = f"^v{main_new_version}$"
    # Rebuild the mapping with the main-only override FIRST. The backport tool applies
    # the first matching branchLabelMapping key, so the specific main override (e.g.
    # ^v9.6.0$ -> main) must precede the generic ^vX.Y.Z$ -> X.Y rule. Otherwise the
    # main version label resolves to a non-existent MAJOR.MINOR release branch and the
    # backport fails (see PR #3071 attempting a non-existent "9.6" branch).
    # Exclude new_main_key so a misconfigured existing entry (e.g. ^v9.6.0$ -> "9.6")
    # cannot overwrite the correct override below via the update() call.
    non_main = {k: v for k, v in old_mapping.items() if v != "main" and k != new_main_key}
    new_mapping: dict[str, str] = {new_main_key: "main"}
    new_mapping.update(non_main)
    # dict equality ignores order, so compare item order explicitly to catch reorders.
    if list(new_mapping.items()) != list(old_mapping.items()):
        changed = True
    data["branchLabelMapping"] = new_mapping

    return changed


def _cmd_update(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_file():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 1

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    try:
        changed = update_backportrc_for_minor_freeze(
            data,
            new_release_branch=args.new_release_branch,
            main_new_version=args.main_new_version,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if not changed:
        print(f"OK: {path} already configured for branch {args.new_release_branch} and main {args.main_new_version}")
        return 0

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")

    print(
        f"Updated {path}: added branch {args.new_release_branch}, "
        f"main label mapping v{args.main_new_version}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Update .backportrc.json for minor freeze")
    parser.add_argument("--path", default=".backportrc.json", help="Path to backportrc file")
    parser.add_argument("--new-release-branch", required=True, help="New release branch (MAJOR.MINOR)")
    parser.add_argument("--main-new-version", required=True, help="New version on main (MAJOR.MINOR.PATCH)")
    args = parser.parse_args()
    return _cmd_update(args)


if __name__ == "__main__":
    sys.exit(main())
