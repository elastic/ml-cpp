#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Guard rails for automatic backport of version-bump automation PRs."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKPORT_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "backport.yml"
_BUMP_MAIN_MINOR_FREEZE = _REPO_ROOT / "dev-tools" / "bump_main_minor_freeze.sh"


def test_backport_workflow_skips_no_backport_label() -> None:
    text = _BACKPORT_WORKFLOW.read_text(encoding="utf-8")
    assert "no-backport" in text
    assert "!(contains(github.event.pull_request.labels.*.name, 'no-backport'))" in text


def test_backport_workflow_skips_minor_freeze_title() -> None:
    text = _BACKPORT_WORKFLOW.read_text(encoding="utf-8")
    assert "!contains(github.event.pull_request.title, '(minor freeze)')" in text


def test_bump_main_minor_freeze_applies_no_backport_label() -> None:
    text = _BUMP_MAIN_MINOR_FREEZE.read_text(encoding="utf-8")
    assert '--label "no-backport"' in text
