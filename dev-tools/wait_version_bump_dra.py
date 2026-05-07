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
ml_cpp_version_bump_changed via ``buildkite-agent meta-data get`` and exits
immediately when no PR was opened.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

POLL_SECONDS = 30
TIMEOUT_SECONDS = 240 * 60

STAGING_TMPL = "https://artifacts-staging.elastic.co/ml-cpp/latest/{branch}.json"
SNAPSHOT_TMPL = "https://storage.googleapis.com/elastic-artifacts-snapshot/ml-cpp/latest/{branch}.json"


def _meta_get(key: str) -> str | None:
    if os.environ.get("BUILDKITE") != "true":
        return None
    try:
        proc = subprocess.run(
            ["buildkite-agent", "meta-data", "get", key],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        v = proc.stdout.strip()
        return v if v else None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


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


def main() -> int:
    if os.environ.get("DRY_RUN") == "true":
        print("DRY_RUN=true — skipping DRA wait.")
        return 0

    if _meta_get("ml_cpp_version_bump_changed") != "true":
        print(
            "ml_cpp_version_bump_changed is not true — no PR opened; skipping DRA wait.",
            file=sys.stderr,
        )
        return 0

    branch = os.environ.get("BRANCH", "").strip()
    new_version = os.environ.get("NEW_VERSION", "").strip()
    if not branch or not new_version:
        print("ERROR: BRANCH and NEW_VERSION must be set.", file=sys.stderr)
        return 1

    staging_url = STAGING_TMPL.format(branch=branch)
    snapshot_url = SNAPSHOT_TMPL.format(branch=branch)
    want_staging = new_version
    want_snapshot = f"{new_version}-SNAPSHOT"

    print(f"Waiting for DRA artifacts (timeout {TIMEOUT_SECONDS}s, poll {POLL_SECONDS}s)...")
    print(f"  staging:  {want_staging!r}  <= {staging_url}")
    print(f"  snapshot: {want_snapshot!r} <= {snapshot_url}")

    deadline = time.monotonic() + TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        st = _fetch_version(staging_url)
        sn = _fetch_version(snapshot_url)
        if st == want_staging and sn == want_snapshot:
            print("OK: staging and snapshot versions matched.")
            return 0
        if st is not None or sn is not None:
            print(f"  staging={st!r} snapshot={sn!r} (still waiting)")
        time.sleep(POLL_SECONDS)

    print("ERROR: timed out waiting for DRA artifact versions.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
