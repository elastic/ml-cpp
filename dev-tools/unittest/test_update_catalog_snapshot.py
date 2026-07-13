#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Pytest tests for dev-tools/update_catalog_snapshot.py.

Verifies a newly cut release branch is registered in the snapshot build pipeline
(filter_condition + daily schedule) with a minimal, comment-preserving diff, that
the operation is idempotent, and that the real catalog-info.yaml stays valid YAML.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_DEV_TOOLS = Path(__file__).resolve().parents[1]
if str(_DEV_TOOLS) not in sys.path:
    sys.path.insert(0, str(_DEV_TOOLS))

import update_catalog_snapshot as ucs  # noqa: E402

_REPO_ROOT = _DEV_TOOLS.parent
_CATALOG = _REPO_ROOT / "catalog-info.yaml"

# A branch that will never actually ship, so real-catalog tests stay stable and
# never flip to a no-op (changed == False) once a plausible version is released.
_SENTINEL_BRANCH = "99.0"

# Minimal fixture mirroring the snapshot pipeline block followed by another
# section, so scoping (only the snapshot block is edited) is exercised.
_FIXTURE = """\
# Declare the snapshot build pipeline
---
apiVersion: "backstage.io/v1alpha1"
kind: "Resource"
metadata:
  name: "ml-cpp-snapshot-builds"
spec:
  implementation:
    spec:
      provider_settings:
        build_branches: true
        filter_condition: build.branch == "main" || build.branch == "9.4" || build.branch == "7.17"
        filter_enabled: true
      schedules:
        Daily 7_17:
          branch: '7.17'
          cronline: 30 04 * * *
          message: Daily SNAPSHOT build for 7.17
        Daily 9.4:
          branch: '9.4'
          cronline: 30 01 * * *
          message: Daily SNAPSHOT build for 9.4
        Daily main:
          branch: main
          cronline: 30 00 * * *
          message: Daily SNAPSHOT build for main
      skip_intermediate_builds: true

# Declare the staging build pipeline
---
metadata:
  name: "ml-cpp-staging-builds"
spec:
  implementation:
    spec:
      provider_settings:
        filter_condition: 'build.branch == "main"'
"""


def test_adds_branch_to_filter_after_main() -> None:
    new_text, changed = ucs.add_release_branch_to_snapshot(_FIXTURE, "9.5")
    assert changed is True
    filter_line = next(
        line for line in new_text.splitlines() if "filter_condition:" in line and "build_branches" not in line
    )
    # 9.5 must appear immediately after main and before 9.4 (newest-first).
    assert 'build.branch == "main" || build.branch == "9.5" || build.branch == "9.4"' in filter_line


def test_adds_schedule_before_main_with_next_free_hour() -> None:
    new_text, _ = ucs.add_release_branch_to_snapshot(_FIXTURE, "9.5")
    lines = new_text.splitlines()
    # New schedule inserted just before "Daily main:".
    keys = [ln.strip().rstrip(":") for ln in lines if ln.strip().startswith("Daily ")]
    assert keys == ["Daily 7_17", "Daily 9.4", "Daily 9.5", "Daily main"]
    # Next free half-past hour is 05 (max existing is 04).
    assert "cronline: 30 05 * * *" in new_text
    assert "message: Daily SNAPSHOT build for 9.5" in new_text
    assert "branch: '9.5'" in new_text


def test_does_not_touch_staging_section() -> None:
    new_text, _ = ucs.add_release_branch_to_snapshot(_FIXTURE, "9.5")
    staging = new_text.split("# Declare the staging build pipeline", 1)[1]
    assert "9.5" not in staging


def test_idempotent() -> None:
    once, changed1 = ucs.add_release_branch_to_snapshot(_FIXTURE, "9.5")
    assert changed1 is True
    twice, changed2 = ucs.add_release_branch_to_snapshot(once, "9.5")
    assert changed2 is False
    assert twice == once


def test_partial_state_filter_only_is_completed() -> None:
    # Branch already in filter but no schedule yet: the schedule must still be added.
    text, _ = ucs.add_release_branch_to_snapshot(_FIXTURE, "9.5")
    # Remove the schedule block we just added, keep the filter entry.
    without_schedule = text.replace(
        "        Daily 9.5:\n"
        "          branch: '9.5'\n"
        "          cronline: 30 05 * * *\n"
        "          message: Daily SNAPSHOT build for 9.5\n",
        "",
    )
    completed, changed = ucs.add_release_branch_to_snapshot(without_schedule, "9.5")
    assert changed is True
    assert "Daily 9.5:" in completed


def test_partial_state_schedule_only_is_completed() -> None:
    # Inverse of the above: branch already scheduled but missing from the filter
    # (e.g. a hand edit). The filter entry must still be added.
    text, _ = ucs.add_release_branch_to_snapshot(_FIXTURE, "9.5")
    # Remove only the filter entry we just added, keep the schedule block.
    without_filter = text.replace(' || build.branch == "9.5"', "")
    assert 'build.branch == "9.5"' not in without_filter
    assert "Daily 9.5:" in without_filter

    completed, changed = ucs.add_release_branch_to_snapshot(without_filter, "9.5")
    assert changed is True
    assert 'build.branch == "9.5"' in completed
    # The schedule must not be duplicated when it was already present.
    assert completed.count("Daily 9.5:") == 1


def test_fallback_appends_after_last_entry_when_no_daily_main() -> None:
    # A schedules block without a "Daily main" entry: the new schedule must be
    # appended after the last existing entry, before the next sibling key
    # (skip_intermediate_builds), not prepended after "schedules:".
    fixture = """\
# Declare the snapshot build pipeline
---
metadata:
  name: "ml-cpp-snapshot-builds"
spec:
  implementation:
    spec:
      provider_settings:
        filter_condition: build.branch == "main" || build.branch == "9.4"
      schedules:
        Daily 9.4:
          branch: '9.4'
          cronline: 30 01 * * *
          message: Daily SNAPSHOT build for 9.4
      skip_intermediate_builds: true

# Declare the staging build pipeline
---
metadata:
  name: "other"
"""
    new_text, changed = ucs.add_release_branch_to_snapshot(fixture, "9.5")
    assert changed is True
    lines = new_text.splitlines()
    keys = [ln.strip().rstrip(":") for ln in lines if ln.strip().startswith("Daily ")]
    assert keys == ["Daily 9.4", "Daily 9.5"]
    # New block sits between the last schedule entry and the sibling key.
    msg_idx = next(i for i, ln in enumerate(lines) if "Daily SNAPSHOT build for 9.5" in ln)
    sib_idx = next(i for i, ln in enumerate(lines) if "skip_intermediate_builds:" in ln)
    assert msg_idx < sib_idx
    yaml = pytest.importorskip("yaml")
    docs = list(yaml.safe_load_all(new_text))
    snap = next(
        d for d in docs if d and d.get("metadata", {}).get("name") == "ml-cpp-snapshot-builds"
    )
    assert snap["spec"]["implementation"]["spec"]["schedules"]["Daily 9.5"]["branch"] == "9.5"


def test_rejects_non_semver_branch() -> None:
    with pytest.raises(ValueError, match="MAJOR.MINOR"):
        ucs.add_release_branch_to_snapshot(_FIXTURE, "main")
    with pytest.raises(ValueError, match="MAJOR.MINOR"):
        ucs.add_release_branch_to_snapshot(_FIXTURE, "9.5.0")


def test_missing_snapshot_anchor_raises() -> None:
    with pytest.raises(ValueError, match="anchor not found"):
        ucs.add_release_branch_to_snapshot("no snapshot pipeline here\n", "9.5")


@pytest.mark.skipif(not _CATALOG.is_file(), reason="catalog-info.yaml not found")
def test_real_catalog_edit_is_valid_yaml_and_unique_crons() -> None:
    yaml = pytest.importorskip("yaml")
    text = _CATALOG.read_text(encoding="utf-8")
    new_text, changed = ucs.add_release_branch_to_snapshot(text, _SENTINEL_BRANCH)
    assert changed is True

    docs = list(yaml.safe_load_all(new_text))
    snap = next(
        d for d in docs if d and d.get("metadata", {}).get("name") == "ml-cpp-snapshot-builds"
    )
    spec = snap["spec"]["implementation"]["spec"]
    assert f'build.branch == "{_SENTINEL_BRANCH}"' in spec["provider_settings"]["filter_condition"]
    assert spec["schedules"][f"Daily {_SENTINEL_BRANCH}"]["branch"] == _SENTINEL_BRANCH

    crons = [s["cronline"] for s in spec["schedules"].values()]
    assert len(crons) == len(set(crons)), f"cronline collision: {crons}"


@pytest.mark.skipif(not _CATALOG.is_file(), reason="catalog-info.yaml not found")
def test_real_catalog_only_snapshot_block_changes() -> None:
    text = _CATALOG.read_text(encoding="utf-8")
    new_text, _ = ucs.add_release_branch_to_snapshot(text, _SENTINEL_BRANCH)
    # The staging section (and everything after it) must be untouched.
    marker = "# Declare the staging build pipeline"
    assert text.split(marker, 1)[1] == new_text.split(marker, 1)[1]
