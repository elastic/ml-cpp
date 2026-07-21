#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Tests for helpers in dev-tools/version_bump_lib.sh."""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIB_SH = _REPO_ROOT / "dev-tools" / "version_bump_lib.sh"


def _run_lib_snippet(snippet: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    script = textwrap.dedent(
        f"""\
        set -euo pipefail
        # shellcheck source=../version_bump_lib.sh
        source "{_LIB_SH}"
        {snippet}
        """
    )
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=run_env,
    )


def test_snapshot_helpers_copies_files_and_is_independent_of_src(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "helper_a.sh").write_text("#!/bin/bash\necho a\n", encoding="utf-8")
    (src / "helper_b.py").write_text("print('b')\n", encoding="utf-8")

    completed = _run_lib_snippet(
        f"""
        dest=$(version_bump_snapshot_helpers "{src}" helper_a.sh helper_b.py)
        printf '%s\\n' "$dest"
        test -f "$dest/helper_a.sh"
        test -f "$dest/helper_b.py"
        test -x "$dest/helper_a.sh"
        # Mutating the source must not affect the snapshot.
        echo mutated > "{src}/helper_a.sh"
        grep -q mutated "$dest/helper_a.sh" && exit 2 || true
        rm -rf "$dest"
        """
    )
    assert completed.returncode == 0, completed.stderr
    dest_line = completed.stdout.strip().splitlines()[-1]
    assert dest_line
    assert not Path(dest_line).exists()


def test_snapshot_helpers_fails_on_missing_file(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "present.sh").write_text("#!/bin/bash\n", encoding="utf-8")

    completed = _run_lib_snippet(
        f'version_bump_snapshot_helpers "{src}" present.sh missing.sh'
    )
    assert completed.returncode != 0
    assert "missing helper to snapshot" in completed.stderr


def test_create_github_pull_request_in_snapshot_finds_ensure_github_cli() -> None:
    """Regression: after checkout, create_pr must resolve ensure_github_cli next to itself."""
    src = _REPO_ROOT / "dev-tools"
    completed = _run_lib_snippet(
        f"""
        dest=$(version_bump_snapshot_helpers "{src}" \\
            create_github_pull_request.sh \\
            ensure_github_cli.sh)
        bash -n "$dest/create_github_pull_request.sh"
        # Snapshot must keep modern --label parsing (missing on older release branches).
        grep -q -- '--label)' "$dest/create_github_pull_request.sh"
        # ensure_github_cli must sit beside it for SCRIPT_DIR lookup.
        test -f "$dest/ensure_github_cli.sh"
        rm -rf "$dest"
        """
    )
    assert completed.returncode == 0, completed.stderr
