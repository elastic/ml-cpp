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
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PIPELINE_PHASE1 = _REPO_ROOT / ".buildkite" / "job-version-bump.json.py"
_PIPELINE_PHASE2 = _REPO_ROOT / ".buildkite" / "job-version-bump-phase2.json.py"
_PIPELINE_PHASE2_MINOR = (
    _REPO_ROOT / ".buildkite" / "job-version-bump-phase2-minor.json.py"
)


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


def _run_phase2_minor(extra_env: dict[str, str] | None = None) -> dict:
    env = os.environ.copy()
    env.pop("VERSION_BUMP_MERGE_AUTO", None)
    env.setdefault("NEW_VERSION", "9.6.0")
    if extra_env:
        env.update(extra_env)
    out = subprocess.check_output(
        [sys.executable, str(_PIPELINE_PHASE2_MINOR)],
        cwd=str(_REPO_ROOT),
        env=env,
        text=True,
    )
    return json.loads(out)


def _run_phase2(extra_env: dict[str, str] | None = None) -> dict:
    env = os.environ.copy()
    env.pop("VERSION_BUMP_MERGE_AUTO", None)
    env.setdefault("NEW_VERSION", "9.5.4")
    if extra_env:
        env.update(extra_env)
    out = subprocess.check_output(
        [sys.executable, str(_PIPELINE_PHASE2)],
        cwd=str(_REPO_ROOT),
        env=env,
        text=True,
    )
    return json.loads(out)


def _run_phase2_raw(
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Like _run_phase2, but for asserting on a failing run.

    _run_phase2 uses check_output, which raises on a non-zero exit instead of
    letting the test inspect it. Used to verify the NEW_VERSION fail-fast guard
    actually fails loudly rather than silently uploading plugins.version="".
    """
    env = os.environ.copy()
    env.pop("VERSION_BUMP_MERGE_AUTO", None)
    env.pop("NEW_VERSION", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(_PIPELINE_PHASE2)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def _run_phase2_minor_raw(
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """_run_phase2_minor's counterpart to _run_phase2_raw, see its docstring."""
    env = os.environ.copy()
    env.pop("VERSION_BUMP_MERGE_AUTO", None)
    env.pop("NEW_VERSION", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(_PIPELINE_PHASE2_MINOR)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


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


def test_phase1_schedule_uses_wolfi_image_like_validate() -> None:
    """Schedule step needs buildkite-agent for meta-data and pipeline upload."""
    pipeline = _run_phase1()
    validate_img = _step_by_key(pipeline, "validate-version-bump")["agents"]["image"]
    sched_img = _step_by_key(pipeline, "schedule-version-bump-follow-up")["agents"][
        "image"
    ]
    assert sched_img == validate_img
    assert "wolfi" in sched_img


def test_phase2_bump_defaults_merge_auto_true() -> None:
    pipeline = _run_phase2()
    bump = _step_by_key(pipeline, "bump-version")
    assert bump["env"]["VERSION_BUMP_MERGE_AUTO"] == "true"


def test_phase2_bump_respects_merge_auto_override_false() -> None:
    pipeline = _run_phase2({"VERSION_BUMP_MERGE_AUTO": "false"})
    bump = _step_by_key(pipeline, "bump-version")
    assert bump["env"]["VERSION_BUMP_MERGE_AUTO"] == "false"


def test_phase2_schedules_dra_wait_check() -> None:
    """The plugin step itself is generated by job-dra-wait.json.py, uploaded only
    at runtime by maybe_schedule_dra_wait.sh once ml_cpp_version_bump_changed is
    known — see test_job_dra_wait_pipeline.py for its plugin config shape."""
    pipeline = _run_phase2({"NEW_VERSION": "9.5.4", "WORKFLOW": "patch"})
    check = _step_by_key(pipeline, "maybe-schedule-dra-wait")
    assert "if" not in check
    assert check["command"] == ["dev-tools/maybe_schedule_dra_wait.sh"]
    assert "fetch-dra-artifacts" not in [s.get("key") for s in pipeline["steps"]]


def test_phase2_dra_skipped_for_sandbox_branch() -> None:
    pipeline = _run_phase2({"BRANCH": "testing-9.5", "NEW_VERSION": "9.5.4"})
    keys = [s.get("key") for s in pipeline["steps"]]
    assert "maybe-schedule-dra-wait" not in keys
    assert "fetch-dra-artifacts" not in keys


def test_phase2_dra_skipped_for_dry_run() -> None:
    """A direct/local generator run under DRY_RUN should omit the check step
    itself, not just rely on maybe_schedule_dra_wait.sh exiting early at
    runtime — see test_maybe_schedule_dra_wait.py for that script's own test."""
    pipeline = _run_phase2({"DRY_RUN": "true", "NEW_VERSION": "9.5.4"})
    keys = [s.get("key") for s in pipeline["steps"]]
    assert "maybe-schedule-dra-wait" not in keys


def test_phase2_dra_sandbox_skip_logs_reason() -> None:
    """The step vanishing silently would look like a broken pipeline; it must explain itself."""
    env = os.environ.copy()
    env.pop("VERSION_BUMP_MERGE_AUTO", None)
    env["BRANCH"] = "testing-9.5"
    env["NEW_VERSION"] = "9.5.4"
    proc = subprocess.run(
        [sys.executable, str(_PIPELINE_PHASE2)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert (
        "Sandbox release branch 'testing-9.5' — not scheduling the Fetch DRA "
        "artifact step" in proc.stderr
    )


def test_phase2_fails_fast_when_new_version_unset() -> None:
    """Generation must fail loudly rather than upload a plugin config with version=''."""
    proc = _run_phase2_raw()
    assert proc.returncode != 0
    assert "NEW_VERSION must be set" in proc.stderr


def test_phase2_order_bump_then_slack_then_dra_check() -> None:
    pipeline = _run_phase2()
    assert (
        _step_by_key(pipeline, "bump-version")["depends_on"]
        == "schedule-version-bump-follow-up"
    )
    assert _step_by_key(pipeline, "queue-slack-notify")["depends_on"] == "bump-version"
    slack_cmd = _step_by_key(pipeline, "queue-slack-notify")["command"]
    assert slack_cmd == [".buildkite/pipelines/send_slack_version_bump_notification.sh"]
    assert (
        _step_by_key(pipeline, "maybe-schedule-dra-wait")["depends_on"]
        == "queue-slack-notify"
    )


def test_phase2_slack_step_uses_same_agent_image_as_bump() -> None:
    """Slack step must run where buildkite-agent is available (see send_slack script)."""
    pipeline = _run_phase2()
    bump_img = _step_by_key(pipeline, "bump-version")["agents"]["image"]
    slack_img = _step_by_key(pipeline, "queue-slack-notify")["agents"]["image"]
    assert slack_img == bump_img


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
            "--body",
            "body",
            "--merge",
            "--merge-auto",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "only one of --merge or --merge-auto" in proc.stderr


def test_create_pr_script_requires_body() -> None:
    """create_github_pull_request.sh requires --body alongside other PR fields."""
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
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "--body" in proc.stderr


def _asserts_label(script: str, label: str) -> bool:
    """True if the script passes ``--label <label>`` in any common quoting style.

    Tolerates double quotes, single quotes, or no quotes so a harmless requote
    of the argument doesn't fail the guard while the behaviour is unchanged.
    """
    pattern = rf"--label\s+(?:\"{re.escape(label)}\"|'{re.escape(label)}'|{re.escape(label)})(?:\s|\))"
    return re.search(pattern, script) is not None


def test_bump_version_pr_applies_required_labels() -> None:
    """Automated bump PRs must open with the labels repo policy requires.

    :ml is the team label, >build is the type label marking a no-changelog
    build change, and ci:skip-es-tests keeps the ES test suite off a pure
    version bump. Both automated PR creators (patch bump and main minor-freeze
    bump) must open compliant instead of needing a human to add them by hand
    (e.g. #3135).
    """
    for name in ("bump_version.sh", "bump_main_minor_freeze.sh"):
        script = (_REPO_ROOT / "dev-tools" / name).read_text()
        for label in ("ci:skip-es-tests", ":ml", ">build"):
            assert _asserts_label(script, label), f"{name}: missing --label {label}"


def test_phase2_minor_has_parallel_freeze_group() -> None:
    pipeline = _run_phase2_minor()
    group = pipeline["steps"][0]
    assert group["key"] == "minor-freeze"
    keys = [s["key"] for s in group["steps"]]
    assert keys == ["create-minor-branch", "bump-main-minor-freeze"]


def test_phase2_minor_order_group_then_slack_then_dra() -> None:
    pipeline = _run_phase2_minor()
    assert _step_by_key(pipeline, "queue-slack-notify")["depends_on"] == "minor-freeze"
    assert (
        _step_by_key(pipeline, "fetch-dra-artifacts")["depends_on"]
        == "queue-slack-notify"
    )


def test_phase2_minor_dra_uses_version_bump_dra_plugin() -> None:
    pipeline = _run_phase2_minor({"NEW_VERSION": "9.6.0"})
    dra = _step_by_key(pipeline, "fetch-dra-artifacts")
    config = dra["plugins"][0]["elastic/version-bump-dra#v1.0.0"]
    assert config == {"product": "ml-cpp", "version": "9.6.0", "workflow": "minor"}


def test_phase2_minor_dra_skipped_for_sandbox_branch() -> None:
    pipeline = _run_phase2_minor({"BRANCH": "testing-9.5", "NEW_VERSION": "9.5.0"})
    keys = [s.get("key") for s in pipeline["steps"]]
    assert "fetch-dra-artifacts" not in keys


def test_phase2_minor_dra_skipped_for_dry_run() -> None:
    """bump_main_minor_freeze.sh never opens a real PR on a DRY_RUN so there is
    nothing for the plugin to wait on; scheduling it anyway would poll real
    production DRA manifests for a dry run."""
    pipeline = _run_phase2_minor({"DRY_RUN": "true", "NEW_VERSION": "9.5.0"})
    keys = [s.get("key") for s in pipeline["steps"]]
    assert "fetch-dra-artifacts" not in keys


def test_phase2_minor_dra_sandbox_skip_logs_reason() -> None:
    """The step vanishing silently would look like a broken pipeline; it must explain itself."""
    env = os.environ.copy()
    env.pop("VERSION_BUMP_MERGE_AUTO", None)
    env["BRANCH"] = "testing-9.5"
    env["NEW_VERSION"] = "9.5.0"
    proc = subprocess.run(
        [sys.executable, str(_PIPELINE_PHASE2_MINOR)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert (
        "Sandbox release branch 'testing-9.5' — not scheduling the Fetch DRA "
        "artifact step" in proc.stderr
    )


def test_phase2_minor_fails_fast_when_new_version_unset() -> None:
    """Generation must fail loudly rather than upload a plugin config with version=''."""
    proc = _run_phase2_minor_raw()
    assert proc.returncode != 0
    assert "NEW_VERSION must be set" in proc.stderr

