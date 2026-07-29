#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Tests for dev-tools/version_bump_upload_phase2.sh."""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UPLOAD_SCRIPT = _REPO_ROOT / "dev-tools" / "version_bump_upload_phase2.sh"


@pytest.fixture
def fake_buildkite_agent(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    count_file = tmp_path / "upload_count"
    count_file.write_text("0")
    agent = bin_dir / "buildkite-agent"
    agent.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/bash
            set -euo pipefail
            if [[ "$1" == "meta-data" && "$2" == "get" ]]; then
                echo "false"
                exit 0
            fi
            if [[ "$1" == "pipeline" && "$2" == "upload" ]]; then
                n=$(cat "{count_file}")
                echo $((n + 1)) > "{count_file}"
                cat > "{tmp_path}/upload-${{n}}.json"
                exit 0
            fi
            echo "unexpected: $*" >&2
            exit 1
            """
        )
    )
    agent.chmod(0o755)
    return bin_dir, count_file


def test_minor_workflow_uploads_phase2_once(fake_buildkite_agent: tuple[Path, Path]) -> None:
    """WORKFLOW=minor must not also upload the patch phase-2 pipeline."""
    bin_dir, count_file = fake_buildkite_agent
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["WORKFLOW"] = "minor"
    env.pop("DRY_RUN", None)

    proc = subprocess.run(
        ["/bin/bash", str(_UPLOAD_SCRIPT)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert count_file.read_text().strip() == "1"

    pipeline = json.loads((count_file.parent / "upload-0.json").read_text())
    assert pipeline["steps"][0]["key"] == "minor-freeze"


def test_patch_workflow_uploads_patch_phase2_once(fake_buildkite_agent: tuple[Path, Path]) -> None:
    bin_dir, count_file = fake_buildkite_agent
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["WORKFLOW"] = "patch"
    env.pop("DRY_RUN", None)

    proc = subprocess.run(
        ["/bin/bash", str(_UPLOAD_SCRIPT)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert count_file.read_text().strip() == "1"

    pipeline = json.loads((count_file.parent / "upload-0.json").read_text())
    assert pipeline["steps"][0]["key"] == "bump-version"
