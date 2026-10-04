#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Tests for .buildkite/pipeline.json.py (the PR build pipeline generator).

Exercises every label/comment code path that uses os.environ so that a missing
'import os' (or similar name-error) is caught before it silently breaks CI on
backport PRs that carry labels like ci:run-qa-tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PIPELINE = _REPO_ROOT / ".buildkite" / "pipeline.json.py"


def _run(extra_env: dict[str, str] | None = None, remove_keys: list[str] | None = None) -> dict:
    """Run pipeline.json.py and return the parsed JSON output."""
    env = os.environ.copy()
    # Strip any real CI env vars that would alter the code path under test.
    for key in (remove_keys or []):
        env.pop(key, None)
    for key in ["GITHUB_PR_TRIGGER_COMMENT", "GITHUB_PR_LABELS",
                "GITHUB_PR_COMMENT_VAR_ACTION", "GITHUB_PR_COMMENT_VAR_ARCH",
                "GITHUB_PR_COMMENT_VAR_PLATFORM", "QAF_TESTS_TO_RUN"]:
        env.pop(key, None)
    if extra_env:
        env.update(extra_env)
    out = subprocess.check_output(
        [sys.executable, str(_PIPELINE)],
        cwd=str(_REPO_ROOT),
        env=env,
        text=True,
    )
    return json.loads(out)


# ---------------------------------------------------------------------------
# Label-triggered paths
# ---------------------------------------------------------------------------

def test_no_labels_builds_all_platforms() -> None:
    pipeline = _run({"GITHUB_PR_LABELS": ""})
    steps = pipeline["steps"]
    labels = [s.get("label", "") for s in steps]
    assert any("Linux" in l for l in labels)
    assert any("MacOS" in l for l in labels)
    assert any("Windows" in l for l in labels)


def test_qa_label_produces_valid_json() -> None:
    """ci:run-qa-tests triggers os.environ.get() — a missing 'import os' crashes here."""
    pipeline = _run({"GITHUB_PR_LABELS": "ci:run-qa-tests"})
    assert isinstance(pipeline, dict)
    assert "steps" in pipeline
    env = pipeline.get("env", {})
    assert "QAF_TESTS_TO_RUN" in env
    assert "ml_cpp_pr" in env["QAF_TESTS_TO_RUN"]


def test_pytorch_label_produces_valid_json() -> None:
    """ci:run-pytorch-tests is the other label that enters the os.environ code path."""
    pipeline = _run({"GITHUB_PR_LABELS": "ci:run-pytorch-tests"})
    assert isinstance(pipeline, dict)
    assert "steps" in pipeline
    env = pipeline.get("env", {})
    assert "QAF_TESTS_TO_RUN" in env
    assert "pytorch_tests" in env["QAF_TESTS_TO_RUN"]


def test_qa_and_pytorch_labels_deduped() -> None:
    pipeline = _run({"GITHUB_PR_LABELS": "ci:run-qa-tests,ci:run-pytorch-tests"})
    env = pipeline.get("env", {})
    suites = env.get("QAF_TESTS_TO_RUN", "").split(",")
    assert suites.count("ml_cpp_pr") == 1
    assert suites.count("pytorch_tests") == 1


def test_qa_label_with_qaf_override_env() -> None:
    """QAF_TESTS_TO_RUN env override is split and forwarded, not appended whole."""
    pipeline = _run({
        "GITHUB_PR_LABELS": "ci:run-qa-tests",
        "QAF_TESTS_TO_RUN": "suite_a,suite_b",
    })
    env = pipeline.get("env", {})
    suites = env.get("QAF_TESTS_TO_RUN", "").split(",")
    assert "suite_a" in suites
    assert "suite_b" in suites
    assert "ml_cpp_pr" not in suites


# ---------------------------------------------------------------------------
# Comment-triggered paths
# ---------------------------------------------------------------------------

def test_comment_run_qa_tests_produces_valid_json() -> None:
    pipeline = _run({
        "GITHUB_PR_TRIGGER_COMMENT": "buildkite run_qa_tests",
        "GITHUB_PR_COMMENT_VAR_ACTION": "run_qa_tests",
    })
    assert isinstance(pipeline, dict)
    assert "steps" in pipeline


def test_comment_run_pytorch_tests_produces_valid_json() -> None:
    pipeline = _run({
        "GITHUB_PR_TRIGGER_COMMENT": "buildkite run_pytorch_tests",
        "GITHUB_PR_COMMENT_VAR_ACTION": "run_pytorch_tests",
    })
    assert isinstance(pipeline, dict)
    assert "steps" in pipeline


# ---------------------------------------------------------------------------
# Output is always valid JSON with required top-level keys
# ---------------------------------------------------------------------------

def test_output_always_has_env_and_steps() -> None:
    for labels in ["", "ci:run-qa-tests", "ci:run-pytorch-tests",
                   "ci:build-linux", "ci:build-macos", "ci:build-windows"]:
        pipeline = _run({"GITHUB_PR_LABELS": labels})
        assert "steps" in pipeline, f"missing 'steps' for labels={labels!r}"
        assert "env" in pipeline, f"missing 'env' for labels={labels!r}"
