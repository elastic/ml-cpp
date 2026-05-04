#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Pytest tests for dev-tools/version_bump_validation.py (Buildkite bump rules)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_DEV_TOOLS = Path(__file__).resolve().parents[1]
if str(_DEV_TOOLS) not in sys.path:
    sys.path.insert(0, str(_DEV_TOOLS))

import version_bump_validation as vbu  # noqa: E402

_REPO_ROOT = _DEV_TOOLS.parent
_VALIDATOR_SCRIPT = _DEV_TOOLS / "validate_version_bump_params.sh"
_MODULE = _DEV_TOOLS / "version_bump_validation.py"


def test_parse_semver_ok() -> None:
    assert vbu.parse_semver("9.5.1") == (9, 5, 1)


def test_parse_semver_rejects() -> None:
    assert vbu.parse_semver("9.5") is None
    assert vbu.parse_semver("v9.5.0") is None
    assert vbu.parse_semver("9.5.0.1") is None


def test_patch_ok_consecutive() -> None:
    vbu.validate_version_bump_params(
        current_version="9.5.0",
        new_version="9.5.1",
        branch="9.5",
        workflow="patch",
    )


def test_patch_ok_noop_same_version() -> None:
    vbu.validate_version_bump_params(
        current_version="9.5.1",
        new_version="9.5.1",
        branch="9.5",
        workflow="patch",
    )


def test_patch_rejects_skip() -> None:
    with pytest.raises(ValueError):
        vbu.validate_version_bump_params(
            current_version="9.5.0",
            new_version="9.5.2",
            branch="9.5",
            workflow="patch",
        )


def test_patch_rejects_wrong_branch_minor() -> None:
    with pytest.raises(ValueError):
        vbu.validate_version_bump_params(
            current_version="9.5.0",
            new_version="9.5.1",
            branch="9.4",
            workflow="patch",
        )


def test_patch_rejects_minor_mismatch() -> None:
    with pytest.raises(ValueError):
        vbu.validate_version_bump_params(
            current_version="9.4.9",
            new_version="9.5.1",
            branch="9.5",
            workflow="patch",
        )


def test_minor_ok() -> None:
    vbu.validate_version_bump_params(
        current_version="9.4.12",
        new_version="9.5.0",
        branch="9.5",
        workflow="minor",
    )


def test_minor_rejects_patch_not_zero() -> None:
    with pytest.raises(ValueError):
        vbu.validate_version_bump_params(
            current_version="9.4.12",
            new_version="9.5.1",
            branch="9.5",
            workflow="minor",
        )


def test_minor_rejects_wrong_increment() -> None:
    with pytest.raises(ValueError):
        vbu.validate_version_bump_params(
            current_version="9.4.12",
            new_version="9.6.0",
            branch="9.6",
            workflow="minor",
        )


def test_invalid_workflow() -> None:
    with pytest.raises(ValueError):
        vbu.validate_workflow_name("major")


def test_cli_validate_patch_ok() -> None:
    rc = subprocess.call(
        [
            sys.executable,
            str(_MODULE),
            "validate",
            "--current",
            "9.5.0",
            "--new",
            "9.5.1",
            "--branch",
            "9.5",
            "--workflow",
            "patch",
        ],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert rc == 0


def test_cli_validate_patch_negative() -> None:
    rc = subprocess.call(
        [
            sys.executable,
            str(_MODULE),
            "validate",
            "--current",
            "9.5.0",
            "--new",
            "9.5.2",
            "--branch",
            "9.5",
            "--workflow",
            "patch",
        ],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert rc != 0


def test_cli_validate_minor_ok() -> None:
    rc = subprocess.call(
        [
            sys.executable,
            str(_MODULE),
            "validate",
            "--current",
            "9.4.8",
            "--new",
            "9.5.0",
            "--branch",
            "9.5",
            "--workflow",
            "minor",
        ],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert rc == 0


@pytest.mark.skipif(
    not _VALIDATOR_SCRIPT.is_file(),
    reason="validate_version_bump_params.sh missing",
)
def test_shell_skip_validation_env() -> None:
    env = os.environ.copy()
    env["SKIP_VERSION_VALIDATION"] = "true"
    env.pop("NEW_VERSION", None)
    env.pop("BRANCH", None)
    out = subprocess.run(
        ["/bin/bash", str(_VALIDATOR_SCRIPT)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert out.returncode == 0, out.stderr + out.stdout
