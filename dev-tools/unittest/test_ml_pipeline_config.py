#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Tests for .buildkite/ml_pipeline/config.py (PR pipeline gating)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILDKITE_DIR = _REPO_ROOT / ".buildkite"
_PIPELINE_JSON = _BUILDKITE_DIR / "pipeline.json.py"

sys.path.insert(0, str(_BUILDKITE_DIR))
import ml_pipeline.config as pipeline_config  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_skip_es_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "GITHUB_PR_LABELS",
        "BUILDKITE_PULL_REQUEST_LABELS",
        "BUILDKITE_BRANCH",
        "GITHUB_PR_TRIGGER_COMMENT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_normalize_buildkite_branch_fork_and_plus_separator() -> None:
    assert (
        pipeline_config.normalize_buildkite_branch("edsavage:ci+ml-cpp-version-bump-9.5-9.5.1")
        == "ci/ml-cpp-version-bump-9.5-9.5.1"
    )
    assert (
        pipeline_config.normalize_buildkite_branch(
            "edsavage+ci/ml-cpp-version-bump-manual-test-9.5.0"
        )
        == "ci/ml-cpp-version-bump-manual-test-9.5.0"
    )
    assert (
        pipeline_config.normalize_buildkite_branch("ci+ml-cpp-minor-freeze-main-9.6.0")
        == "ci/ml-cpp-minor-freeze-main-9.6.0"
    )
    assert (
        pipeline_config.normalize_buildkite_branch("ci/ml-cpp-version-bump-9.5-9.5.1")
        == "ci/ml-cpp-version-bump-9.5-9.5.1"
    )


@pytest.mark.parametrize(
    "branch",
    [
        "ci/ml-cpp-version-bump-9.5-9.5.1",
        "edsavage:ci/ml-cpp-minor-freeze-main-9.6.0-bk42",
        "ci+ml-cpp-minor-freeze-main-9.6.0",
    ],
)
def test_is_version_bump_topic_branch_matches_automation_branches(branch: str) -> None:
    assert pipeline_config.is_version_bump_topic_branch(branch)


def test_is_version_bump_topic_branch_rejects_feature_branches() -> None:
    assert not pipeline_config.is_version_bump_topic_branch("feature/minor-version-bump")


def test_skip_es_tests_from_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_PR_LABELS", ":ml,ci:skip-es-tests")
    config = pipeline_config.Config()
    config.parse()
    assert config.skip_es_tests is True


def test_skip_es_tests_from_topic_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "BUILDKITE_BRANCH",
        "edsavage+ci/ml-cpp-version-bump-manual-test-9.5.0",
    )
    config = pipeline_config.Config()
    config.parse()
    assert config.skip_es_tests is True


def test_skip_es_tests_from_github_pr_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_PR_BRANCH", "ci/ml-cpp-version-bump-manual-test-9.5.0")
    config = pipeline_config.Config()
    config.parse()
    assert config.skip_es_tests is True


def test_skip_es_tests_false_for_normal_pr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_PR_LABELS", ":ml,>enhancement")
    monkeypatch.setenv("BUILDKITE_BRANCH", "edsavage:feature/my-change")
    config = pipeline_config.Config()
    config.parse()
    assert config.skip_es_tests is False


def test_pipeline_json_omits_es_test_upload_steps_when_skip_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_PR_LABELS", "ci:skip-es-tests")
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, str(_PIPELINE_JSON)],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=env,
    )
    pipeline = json.loads(proc.stdout)
    labels = [step.get("label", "") for step in pipeline["steps"]]
    assert not any("ES tests" in label for label in labels)
    assert not any("Inference Integration Tests" in label for label in labels)


def test_pipeline_json_includes_es_test_upload_steps_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_PR_LABELS", raising=False)
    monkeypatch.delenv("BUILDKITE_BRANCH", raising=False)
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_PR_TRIGGER_COMMENT"}
    proc = subprocess.run(
        [sys.executable, str(_PIPELINE_JSON)],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=env,
    )
    pipeline = json.loads(proc.stdout)
    labels = [step.get("label", "") for step in pipeline["steps"]]
    assert any("ES tests x86_64" in label for label in labels)
    assert any("ES tests aarch64" in label for label in labels)
