#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Poll DRA staging/snapshot JSON until versions match (replaces json-watcher plugin).

Buildkite step conditionals cannot use build meta-data; this script reads
ml_cpp_version_bump_noop / ml_cpp_version_bump_changed via ``buildkite-agent
meta-data get`` and exits immediately when the wait is not needed.

Patch (WORKFLOW=patch): waits for staging + snapshot on BRANCH at NEW_VERSION.

Minor (WORKFLOW=minor): waits for three artifact sets after feature freeze:
  - snapshot on main at MAIN_NEW_VERSION-SNAPSHOT
  - snapshot + staging on release branch BRANCH at NEW_VERSION
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import version_bump_validation as vbu

POLL_SECONDS = 30
TIMEOUT_SECONDS = 240 * 60
PROGRESS_LOG_EVERY = 1

STAGING_TMPL = "https://artifacts-staging.elastic.co/ml-cpp/latest/{branch}.json"
SNAPSHOT_TMPL = "https://storage.googleapis.com/elastic-artifacts-snapshot/ml-cpp/latest/{branch}.json"


def _meta_get(key: str) -> str | None:
    """Read Buildkite meta-data. Returns None when not on Buildkite or key is unset."""
    if os.environ.get("BUILDKITE") != "true":
        return None
    try:
        proc = subprocess.run(
            ["buildkite-agent", "meta-data", "get", key],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError:
        print(
            "ERROR: BUILDKITE=true but buildkite-agent is not available; "
            "cannot read meta-data for DRA wait gating.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    except subprocess.TimeoutExpired:
        print(
            f"ERROR: buildkite-agent meta-data get {key!r} timed out.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None

    err = (proc.stderr or "").strip()
    out = (proc.stdout or "").strip()

    if proc.returncode == 0:
        return out if out else None

    err_lower = err.lower()
    if proc.returncode == 1 and (
        "not found" in err_lower
        or "does not exist" in err_lower
        or "couldn't find" in err_lower
        or "could not find" in err_lower
    ):
        return None

    print(
        f"ERROR: buildkite-agent meta-data get {key!r} failed "
        f"(exit {proc.returncode}): {err or '(no stderr)'}",
        file=sys.stderr,
    )
    raise SystemExit(2) from None


def _fetch_version(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ml-cpp-version-bump-dra-wait"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ver = data.get("version")
        if ver is None:
            return None
        return str(ver).strip()
    except (urllib.error.URLError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None


def _wait_for_checks(checks: list[tuple[str, str, str]]) -> int:
    """Poll until all (label, url, expected_version) match."""
    print(f"Waiting for DRA artifacts (timeout {TIMEOUT_SECONDS}s, poll {POLL_SECONDS}s)...")
    for label, url, expected in checks:
        print(f"  {label}: {expected!r} <= {url}")

    deadline = time.monotonic() + TIMEOUT_SECONDS
    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        pending = []
        for label, url, expected in checks:
            got = _fetch_version(url)
            if got != expected:
                pending.append(f"{label}={got!r}")
        if not pending:
            print("OK: all DRA artifact versions matched.")
            return 0
        if iteration % PROGRESS_LOG_EVERY == 0:
            print(f"  still waiting: {', '.join(pending)}")
        time.sleep(POLL_SECONDS)

    print("ERROR: timed out waiting for DRA artifact versions.", file=sys.stderr)
    return 1


def _wait_patch(branch: str, new_version: str) -> int:
    staging_url = STAGING_TMPL.format(branch=branch)
    snapshot_url = SNAPSHOT_TMPL.format(branch=branch)
    checks = [
        ("staging", staging_url, new_version),
        ("snapshot", snapshot_url, f"{new_version}-SNAPSHOT"),
    ]
    return _wait_for_checks(checks)


def _wait_minor(branch: str, new_version: str, main_new_version: str) -> int:
    main_snapshot_url = SNAPSHOT_TMPL.format(branch="main")
    branch_staging_url = STAGING_TMPL.format(branch=branch)
    branch_snapshot_url = SNAPSHOT_TMPL.format(branch=branch)
    checks = [
        ("main snapshot", main_snapshot_url, f"{main_new_version}-SNAPSHOT"),
        ("release snapshot", branch_snapshot_url, f"{new_version}-SNAPSHOT"),
        ("release staging", branch_staging_url, new_version),
    ]
    return _wait_for_checks(checks)


def main() -> int:
    if os.environ.get("DRY_RUN") == "true":
        print("DRY_RUN=true — skipping DRA wait.")
        return 0

    if _meta_get("ml_cpp_version_bump_noop") == "true":
        print(
            "ml_cpp_version_bump_noop is true — nothing to wait for; skipping DRA wait.",
            file=sys.stderr,
        )
        return 0

    workflow = os.environ.get("WORKFLOW", "patch").strip().lower()
    branch = os.environ.get("BRANCH", "").strip()
    new_version = os.environ.get("NEW_VERSION", "").strip()
    if not branch or not new_version:
        print("ERROR: BRANCH and NEW_VERSION must be set.", file=sys.stderr)
        return 1

    if vbu.is_sandbox_release_branch(branch):
        print(
            f"Sandbox release branch {branch!r} — skipping DRA wait "
            f"(no artifacts published for {vbu.SANDBOX_BRANCH_PREFIX}* refs).",
            file=sys.stderr,
        )
        return 0

    if workflow == "minor":
        main_new_version = _meta_get("ml_cpp_version_bump_main_new_version")
        if not main_new_version:
            main_new_version = os.environ.get("MAIN_NEW_VERSION", "").strip()
        if not main_new_version:
            try:
                main_new_version = vbu.derive_main_new_version(new_version)
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
        print(
            f"Minor freeze DRA wait: release branch {branch} @ {new_version}, "
            f"main @ {main_new_version}"
        )
        return _wait_minor(branch, new_version, main_new_version)

    if _meta_get("ml_cpp_version_bump_changed") != "true":
        print(
            "ml_cpp_version_bump_changed is not true — no PR opened; skipping DRA wait.",
            file=sys.stderr,
        )
        return 0

    return _wait_patch(branch, new_version)


if __name__ == "__main__":
    sys.exit(main())
