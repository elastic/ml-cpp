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

    with (
        patch.dict(
            "os.environ",
            {
                "BRANCH": "9.5",
                "NEW_VERSION": "9.5.1",
                "BUILDKITE": "false",
            },
            clear=False,
        ),
        patch.object(mod, "_meta_get", return_value="true"),
        patch.object(mod, "_fetch_version", return_value=None),
        patch.object(mod.time, "monotonic", side_effect=fake_monotonic),
        patch.object(mod.time, "sleep", side_effect=advance_sleep),
        patch.object(mod, "TIMEOUT_SECONDS", 120),
        patch.object(mod, "POLL_SECONDS", 1),
        patch.object(mod, "PROGRESS_LOG_EVERY", 1),
    ):
        assert mod.main() == 1

    out = capsys.readouterr().out
    assert "staging=None snapshot=None (still waiting)" in out
