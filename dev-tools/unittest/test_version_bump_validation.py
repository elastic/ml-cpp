#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Pytest tests for dev-tools/version_bump_validation.py (Buildkite bump rules).

Integration tests (real ``git fetch`` + ``validate_version_bump_params.sh``) are
opt-in so CI stays deterministic:

    export VERSION_BUMP_GIT_INTEGRATION=1
    export VERSION_BUMP_TEST_BRANCH=9.5   # MAJOR.MINOR branch that exists on origin
    python3 -m pip install -r dev-tools/test-requirements.txt
    ./dev-tools/run_dev_tools_tests.sh

Optional: ``VERSION_BUMP_SKIP_NEGATIVE_INTEGRATION=1`` to skip the negative
``patch+2`` check only.
"""

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
    )


def test_patch_ok_noop_same_version() -> None:
    vbu.validate_version_bump_params(
        current_version="9.5.1",
        new_version="9.5.1",
        branch="9.5",
    )


def test_patch_rejects_skip() -> None:
    with pytest.raises(ValueError):
        vbu.validate_version_bump_params(
            current_version="9.5.0",
            new_version="9.5.2",
            branch="9.5",
        )


def test_patch_rejects_wrong_release_branch() -> None:
    with pytest.raises(ValueError):
        vbu.validate_version_bump_params(
            current_version="9.5.0",
            new_version="9.5.1",
            branch="9.4",
        )


def test_patch_rejects_major_minor_mismatch() -> None:
    with pytest.raises(ValueError):
        vbu.validate_version_bump_params(
            current_version="9.4.9",
            new_version="9.5.1",
            branch="9.5",
        )


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
        ],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert rc != 0


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


@pytest.mark.skipif(
    not _VALIDATOR_SCRIPT.is_file(),
    reason="validate_version_bump_params.sh missing",
)
def test_shell_rejects_non_patch_workflow() -> None:
    """Upstream may send WORKFLOW=minor; fail before git fetch."""
    env = os.environ.copy()
    env["WORKFLOW"] = "minor"
    env["NEW_VERSION"] = "9.5.1"
    env["BRANCH"] = "9.5"
    env.pop("SKIP_VERSION_VALIDATION", None)
    out = subprocess.run(
        ["/bin/bash", str(_VALIDATOR_SCRIPT)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert out.returncode != 0, out.stderr + out.stdout
    assert "WORKFLOW" in out.stderr or "WORKFLOW" in out.stdout


def _integration_requested() -> bool:
    return os.environ.get("VERSION_BUMP_GIT_INTEGRATION") == "1"


def _integration_branch() -> str | None:
    b = os.environ.get("VERSION_BUMP_TEST_BRANCH", "").strip()
    return b or None


def _read_version_from_fetch_head(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "show", "FETCH_HEAD:gradle.properties"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"git show FETCH_HEAD:gradle.properties failed: {proc.stderr}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith("elasticsearchVersion="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("elasticsearchVersion not found in FETCH_HEAD gradle.properties")


@pytest.fixture
def git_patch_integration_branch() -> str:
    """Release branch MAJOR.MINOR; requires network + origin ref."""
    if not _integration_requested():
        pytest.skip(
            "Set VERSION_BUMP_GIT_INTEGRATION=1 and VERSION_BUMP_TEST_BRANCH "
            "(e.g. 9.5) to run git integration tests."
        )
    br = _integration_branch()
    if not br:
        pytest.skip("VERSION_BUMP_TEST_BRANCH is not set.")
    return br


@pytest.mark.integration
@pytest.mark.skipif(
    not _VALIDATOR_SCRIPT.is_file(),
    reason="validate_version_bump_params.sh missing",
)
def test_integration_patch_validate_script_with_git_fetch(git_patch_integration_branch: str) -> None:
    """Run validate_version_bump_params.sh after fetch; NEW_VERSION = patch+1 from origin."""
    branch = git_patch_integration_branch
    fetch = subprocess.run(
        ["git", "fetch", "origin", branch],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert fetch.returncode == 0, fetch.stderr + fetch.stdout

    cur = _read_version_from_fetch_head(_REPO_ROOT)
    triple = vbu.parse_semver(cur)
    assert triple is not None, f"unexpected elasticsearchVersion on branch: {cur!r}"
    maj, mino, pat = triple
    new_version = f"{maj}.{mino}.{pat + 1}"

    env = os.environ.copy()
    env["NEW_VERSION"] = new_version
    env["BRANCH"] = branch
    env["WORKFLOW"] = "patch"
    env.pop("SKIP_VERSION_VALIDATION", None)

    out = subprocess.run(
        ["/bin/bash", str(_VALIDATOR_SCRIPT)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr + out.stdout


@pytest.mark.integration
@pytest.mark.skipif(
    not _VALIDATOR_SCRIPT.is_file(),
    reason="validate_version_bump_params.sh missing",
)
@pytest.mark.skipif(
    os.environ.get("VERSION_BUMP_SKIP_NEGATIVE_INTEGRATION") == "1",
    reason="VERSION_BUMP_SKIP_NEGATIVE_INTEGRATION=1",
)
def test_integration_patch_validate_script_rejects_bad_jump(git_patch_integration_branch: str) -> None:
    """Same fetch as production path; NEW_VERSION = patch+2 must fail validation."""
    branch = git_patch_integration_branch
    fetch = subprocess.run(
        ["git", "fetch", "origin", branch],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert fetch.returncode == 0, fetch.stderr + fetch.stdout

    cur = _read_version_from_fetch_head(_REPO_ROOT)
    triple = vbu.parse_semver(cur)
    assert triple is not None
    maj, mino, pat = triple
    bad_version = f"{maj}.{mino}.{pat + 2}"

    env = os.environ.copy()
    env["NEW_VERSION"] = bad_version
    env["BRANCH"] = branch
    env["WORKFLOW"] = "patch"
    env.pop("SKIP_VERSION_VALIDATION", None)

    out = subprocess.run(
        ["/bin/bash", str(_VALIDATOR_SCRIPT)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode != 0, "validator should reject non-consecutive patch bump"
