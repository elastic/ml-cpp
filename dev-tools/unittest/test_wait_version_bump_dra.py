#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Tests for dev-tools/wait_version_bump_dra.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WAIT_SCRIPT = _REPO_ROOT / "dev-tools" / "wait_version_bump_dra.py"


def _load_wait_module():
    spec = importlib.util.spec_from_file_location("wait_version_bump_dra", _WAIT_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wait_version_bump_dra"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_poll_logs_progress_when_versions_unavailable(capsys) -> None:
    """Log each poll even when staging/snapshot fetches return None (not stalled)."""
    mod = _load_wait_module()
    t = 0.0

    def fake_monotonic() -> float:
        return t

    def advance_sleep(_seconds: float) -> None:
        nonlocal t
        t += float(_seconds) + 1.0

    def meta_side_effect(key: str) -> str | None:
        if key == "ml_cpp_version_bump_noop":
            return None
        if key == "ml_cpp_version_bump_changed":
            return "true"
        return None

    with (
        patch.dict(
            "os.environ",
            {
                "BRANCH": "9.5",
                "NEW_VERSION": "9.5.1",
                "WORKFLOW": "patch",
                "BUILDKITE": "false",
            },
            clear=False,
        ),
        patch.object(mod, "_meta_get", side_effect=meta_side_effect),
        patch.object(mod, "_fetch_version", return_value=None),
        patch.object(mod.time, "monotonic", side_effect=fake_monotonic),
        patch.object(mod.time, "sleep", side_effect=advance_sleep),
        patch.object(mod, "TIMEOUT_SECONDS", 120),
        patch.object(mod, "POLL_SECONDS", 1),
        patch.object(mod, "PROGRESS_LOG_EVERY", 1),
    ):
        assert mod.main() == 1

    out = capsys.readouterr().out
    assert "still waiting: staging=None, snapshot=None" in out


def test_wait_minor_polls_version_keyed_alias_for_main_snapshot() -> None:
    """release-manager's project-configs dir for main is "master", not "main", so it only
    ever publishes the branch-keyed "latest" snapshot alias as .../latest/master.json —
    .../latest/main.json is never created. _wait_minor must poll the version-keyed alias
    (.../latest/{version}.json) for the main snapshot check instead, sidestepping the
    branch/master naming mismatch entirely."""
    mod = _load_wait_module()
    captured: list[tuple[str, str, str]] = []

    def fake_wait_for_checks(checks: list[tuple[str, str, str]]) -> int:
        captured.extend(checks)
        return 0

    with patch.object(mod, "_wait_for_checks", side_effect=fake_wait_for_checks):
        assert mod._wait_minor("9.5", "9.5.0", "9.6.0") == 0

    main_snapshot = next(c for c in captured if c[0] == "main snapshot")
    _, url, expected = main_snapshot
    assert url == "https://storage.googleapis.com/elastic-artifacts-snapshot/ml-cpp/latest/9.6.0-SNAPSHOT.json"
    assert "latest/main.json" not in url
    assert "latest/master.json" not in url
    assert expected == "9.6.0-SNAPSHOT"


def test_main_skips_dra_wait_for_sandbox_branch(capsys) -> None:
    mod = _load_wait_module()
    with patch.dict(
        "os.environ",
        {
            "BRANCH": "testing-9.5",
            "NEW_VERSION": "9.5.0",
            "WORKFLOW": "minor",
            "BUILDKITE": "false",
        },
        clear=False,
    ):
        assert mod.main() == 0

    err = capsys.readouterr().err
    assert "Sandbox release branch" in err
    assert "testing-9.5" in err
