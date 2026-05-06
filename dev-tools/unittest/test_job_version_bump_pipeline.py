#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Tests for .buildkite/job-version-bump.json.py pipeline generator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PIPELINE_SCRIPT = _REPO_ROOT / ".buildkite" / "job-version-bump.json.py"


def _run_pipeline_generator(extra_env: dict[str, str] | None = None) -> dict:
    """Run the generator with a clean env (drops inherited VERSION_BUMP_MERGE_AUTO unless set)."""
    env = os.environ.copy()
    env.pop("VERSION_BUMP_MERGE_AUTO", None)
    if extra_env:
        env.update(extra_env)
    out = subprocess.check_output(
        [sys.executable, str(_PIPELINE_SCRIPT)],
        cwd=str(_REPO_ROOT),
        env=env,
        text=True,
    )
    return json.loads(out)


def _bump_step(pipeline: dict) -> dict:
    steps = pipeline["steps"]
    bump = next(s for s in steps if s.get("key") == "bump-version")
    return bump


def _dra_step(pipeline: dict) -> dict:
    steps = pipeline["steps"]
    return next(s for s in steps if s.get("key") == "fetch-dra-artifacts")


def test_bump_step_defaults_merge_auto_true() -> None:
    pipeline = _run_pipeline_generator()
    assert _bump_step(pipeline)["env"]["VERSION_BUMP_MERGE_AUTO"] == "true"


def test_bump_step_respects_merge_auto_override_false() -> None:
    pipeline = _run_pipeline_generator({"VERSION_BUMP_MERGE_AUTO": "false"})
    assert _bump_step(pipeline)["env"]["VERSION_BUMP_MERGE_AUTO"] == "false"


def test_dra_step_requires_bump_meta_and_not_dry_run() -> None:
    pipeline = _run_pipeline_generator()
    cond = _dra_step(pipeline)["if"]
    assert 'build.env("DRY_RUN") != "true"' in cond
    assert 'build.meta_data("ml_cpp_version_bump_changed") == "true"' in cond


def test_mutually_exclusive_merge_flags_script() -> None:
    """create_github_pull_request.sh rejects --merge and --merge-auto together."""
    script = _REPO_ROOT / "dev-tools" / "create_github_pull_request.sh"
    proc = subprocess.run(
        ["bash", str(script), "--repo", "r/r", "--base", "b", "--head", "h", "--title", "t", "--merge", "--merge-auto"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "only one of --merge or --merge-auto" in proc.stderr
