#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Tests for .buildkite/job-dra-wait.json.py (uploaded by maybe_schedule_dra_wait.sh)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PIPELINE_DRA_WAIT = _REPO_ROOT / ".buildkite" / "job-dra-wait.json.py"


def _run_dra_wait(extra_env: dict[str, str] | None = None) -> dict:
    env = os.environ.copy()
    env.setdefault("NEW_VERSION", "9.5.4")
    if extra_env:
        env.update(extra_env)
    out = subprocess.check_output(
        [sys.executable, str(_PIPELINE_DRA_WAIT)],
        cwd=str(_REPO_ROOT),
        env=env,
        text=True,
    )
    return json.loads(out)


def test_dra_wait_uses_version_bump_dra_plugin() -> None:
    pipeline = _run_dra_wait({"NEW_VERSION": "9.5.4", "WORKFLOW": "patch"})
    dra = pipeline["steps"][0]
    assert dra["key"] == "fetch-dra-artifacts"
    assert dra["depends_on"] == "maybe-schedule-dra-wait"
    assert "if" not in dra
    assert "command" not in dra
    config = dra["plugins"][0]["elastic/version-bump-dra#v1.0.0"]
    assert config == {"product": "ml-cpp", "version": "9.5.4", "workflow": "patch"}


def test_dra_wait_defaults_workflow_to_patch() -> None:
    pipeline = _run_dra_wait({"NEW_VERSION": "9.5.4"})
    config = pipeline["steps"][0]["plugins"][0]["elastic/version-bump-dra#v1.0.0"]
    assert config["workflow"] == "patch"


def test_dra_wait_includes_ephemeral_storage() -> None:
    pipeline = _run_dra_wait()
    assert pipeline["steps"][0]["agents"]["ephemeralStorage"] == "1Gi"


def test_dra_wait_fails_fast_when_new_version_unset() -> None:
    env = os.environ.copy()
    env.pop("NEW_VERSION", None)
    proc = subprocess.run(
        [sys.executable, str(_PIPELINE_DRA_WAIT)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "NEW_VERSION must be set" in proc.stderr
