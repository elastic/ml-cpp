#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Tests for .buildkite/job-version-bump*.json.py pipeline generators."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PIPELINE_PHASE1 = _REPO_ROOT / ".buildkite" / "job-version-bump.json.py"
_PIPELINE_PHASE2 = _REPO_ROOT / ".buildkite" / "job-version-bump-phase2.json.py"


def _run_phase1(extra_env: dict[str, str] | None = None) -> dict:
    env = os.environ.copy()
    env.pop("VERSION_BUMP_MERGE_AUTO", None)
    if extra_env:
        env.update(extra_env)
    out = subprocess.check_output(
        [sys.executable, str(_PIPELINE_PHASE1)],
        cwd=str(_REPO_ROOT),
        env=env,
        text=True,
    )
    return json.loads(out)


def _run_phase2(extra_env: dict[str, str] | None = None) -> dict:
    env = os.environ.copy()
    env.pop("VERSION_BUMP_MERGE_AUTO", None)
    if extra_env:
        env.update(extra_env)
    out = subprocess.check_output(
        [sys.executable, str(_PIPELINE_PHASE2)],
        cwd=str(_REPO_ROOT),
        env=env,
        text=True,
    )
    return json.loads(out)


def _step_by_key(pipeline: dict, key: str) -> dict:
    return next(s for s in pipeline["steps"] if s.get("key") == key)


def test_phase1_has_validate_and_schedule_only() -> None:
    pipeline = _run_phase1()
    keys = [s.get("key") for s in pipeline["steps"]]
    assert keys == ["validate-version-bump", "schedule-version-bump-follow-up"]


def test_phase1_has_no_step_if_using_meta_data() -> None:
    """Buildkite rejects build.meta_data in step if expressions at pipeline upload."""
    pipeline = _run_phase1()
    for step in pipeline["steps"]:
        cond = step.get("if")
        if cond is None:
            continue
        assert "build.meta_data" not in cond


def test_phase1_schedule_depends_on_validate() -> None:
    pipeline = _run_phase1()
    sched = _step_by_key(pipeline, "schedule-version-bump-follow-up")
    assert sched["depends_on"] == "validate-version-bump"
    assert sched["command"] == ["dev-tools/version_bump_upload_phase2.sh"]


def test_phase2_bump_defaults_merge_auto_true() -> None:
    pipeline = _run_phase2()
    bump = _step_by_key(pipeline, "bump-version")
    assert bump["env"]["VERSION_BUMP_MERGE_AUTO"] == "true"


def test_phase2_bump_respects_merge_auto_override_false() -> None:
    pipeline = _run_phase2({"VERSION_BUMP_MERGE_AUTO": "false"})
    bump = _step_by_key(pipeline, "bump-version")
    assert bump["env"]["VERSION_BUMP_MERGE_AUTO"] == "false"


def test_phase2_dra_uses_wait_script_not_meta_in_if() -> None:
    pipeline = _run_phase2()
    dra = _step_by_key(pipeline, "fetch-dra-artifacts")
    assert "if" not in dra
    assert "plugins" not in dra
    assert dra["command"] == ["python3", "dev-tools/wait_version_bump_dra.py"]


def test_phase2_slack_depends_on_schedule_key() -> None:
    pipeline = _run_phase2()
    slack = _step_by_key(pipeline, "queue-slack-notify")
    assert slack["depends_on"] == "schedule-version-bump-follow-up"


def test_phase2_bump_depends_on_slack() -> None:
    pipeline = _run_phase2()
    bump = _step_by_key(pipeline, "bump-version")
    assert bump["depends_on"] == "queue-slack-notify"


def test_mutually_exclusive_merge_flags_script() -> None:
    """create_github_pull_request.sh rejects --merge and --merge-auto together."""
    script = _REPO_ROOT / "dev-tools" / "create_github_pull_request.sh"
    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--repo",
            "r/r",
            "--base",
            "b",
            "--head",
            "h",
            "--title",
            "t",
            "--merge",
            "--merge-auto",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "only one of --merge or --merge-auto" in proc.stderr
