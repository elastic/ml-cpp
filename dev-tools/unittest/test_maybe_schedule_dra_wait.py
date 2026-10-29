#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Tests for dev-tools/maybe_schedule_dra_wait.sh."""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "dev-tools" / "maybe_schedule_dra_wait.sh"


def _fake_buildkite_agent(tmp_path: Path, changed_value: str | None) -> Path:
    """changed_value=None simulates the meta-data key never having been set."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    upload_file = tmp_path / "uploaded.json"
    agent = bin_dir / "buildkite-agent"

    if changed_value is None:
        meta_data_body = 'echo "not found" >&2; exit 1'
    else:
        meta_data_body = f'echo "{changed_value}"; exit 0'

    agent.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            set -euo pipefail
            if [[ "$1" == "meta-data" && "$2" == "get" && "$3" == "ml_cpp_version_bump_changed" ]]; then
                {meta_data_body}
            fi
            if [[ "$1" == "pipeline" && "$2" == "upload" ]]; then
                cat > "{upload_file}"
                exit 0
            fi
            echo "unexpected: $*" >&2
            exit 1
            """
        )
    )
    agent.chmod(0o755)
    return bin_dir


def _run(tmp_path: Path, changed_value: str | None, extra_env: dict[str, str] | None = None):
    bin_dir = _fake_buildkite_agent(tmp_path, changed_value)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env.setdefault("NEW_VERSION", "9.5.4")
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        ["/bin/bash", str(_SCRIPT)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc, tmp_path / "uploaded.json"


def test_schedules_dra_wait_when_version_bump_changed(tmp_path: Path) -> None:
    proc, upload_file = _run(tmp_path, "true")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert upload_file.exists()
    pipeline = json.loads(upload_file.read_text())
    assert pipeline["steps"][0]["key"] == "fetch-dra-artifacts"


def test_skips_dra_wait_when_version_bump_unchanged(tmp_path: Path) -> None:
    proc, upload_file = _run(tmp_path, "false")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert not upload_file.exists()
    assert "no PR opened; skipping DRA wait" in proc.stderr


def test_skips_dra_wait_when_version_bump_changed_meta_data_missing(tmp_path: Path) -> None:
    """bump_version.sh always sets this meta-data key, but a missing key must be
    treated the same as false rather than defaulting to scheduling the wait."""
    proc, upload_file = _run(tmp_path, None)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert not upload_file.exists()
    assert "no PR opened; skipping DRA wait" in proc.stderr


def test_skips_dra_wait_when_dry_run(tmp_path: Path) -> None:
    """DRY_RUN must short-circuit before the meta-data check — bump_version.sh
    simulates ml_cpp_version_bump_changed=true under DRY_RUN, which would
    otherwise schedule a real poll against production DRA manifests."""
    proc, upload_file = _run(tmp_path, "true", extra_env={"DRY_RUN": "true"})
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert not upload_file.exists()
    assert "DRY_RUN=true" in proc.stdout
