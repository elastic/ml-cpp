#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Guard rails for automatic backport of version-bump automation PRs.

Assertions use whitespace/quote-tolerant regexes so cosmetic edits to backport.yml
(e.g. !(contains(...)) vs !contains(...), or single vs double quotes) do not cause
noisy failures without an actual behavioral regression.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKPORT_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "backport.yml"
_BUMP_MAIN_MINOR_FREEZE = _REPO_ROOT / "dev-tools" / "bump_main_minor_freeze.sh"


def test_backport_workflow_skips_no_backport_label() -> None:
    text = _BACKPORT_WORKFLOW.read_text(encoding="utf-8")
    # Negated contains() over the PR labels for the no-backport label, tolerating
    # optional wrapping parentheses and either quote style.
    pattern = re.compile(
        r"!\s*\(?\s*contains\(\s*github\.event\.pull_request\.labels\.\*\.name\s*,"
        r"\s*['\"]no-backport['\"]\s*\)",
    )
    assert pattern.search(text), text


def test_backport_workflow_skips_minor_freeze_branch() -> None:
    text = _BACKPORT_WORKFLOW.read_text(encoding="utf-8")
    # Negated startsWith() over the PR head ref for the minor-freeze topic branch,
    # tolerating optional wrapping parentheses and either quote style.
    pattern = re.compile(
        r"!\s*\(?\s*startsWith\(\s*github\.event\.pull_request\.head\.ref\s*,"
        r"\s*['\"]ci/ml-cpp-minor-freeze-main-['\"]\s*\)",
    )
    assert pattern.search(text), text


def test_bump_main_minor_freeze_applies_no_backport_label() -> None:
    text = _BUMP_MAIN_MINOR_FREEZE.read_text(encoding="utf-8")
    # Tolerate quote style and surrounding whitespace on the --label argument.
    pattern = re.compile(r"--label\s+['\"]no-backport['\"]")
    assert pattern.search(text), text


def test_backport_workflow_treats_source_branch_only_as_success() -> None:
    """A current-version label (e.g. v9.6.0 -> main) resolves only to the source
    branch, so the no-branches-exception it produces must be treated as success
    rather than a failing Backport check.
    """
    text = _BACKPORT_WORKFLOW.read_text(encoding="utf-8")
    # The exemption keys off isSourceBranch being exclusively true: it must inspect
    # both the true and false variants (the latter negated) so a real, different
    # target branch is not silently exempted.
    assert re.search(r'"isSourceBranch":\[\[:space:\]\]\*true', text), text
    assert re.search(r'"isSourceBranch":\[\[:space:\]\]\*false', text), text
    # A negated grep (! grep ...) guards the false variant.
    assert re.search(r'!\s*grep[^\n]*isSourceBranch[^\n]*false', text), text
    # The exemption must appear before the version-label hard-failure error, and
    # exit 0 rather than exit 1.
    exempt_idx = text.find("resolved only to the source branch")
    error_idx = text.find("no-branches-exception while this PR had version labels")
    assert exempt_idx != -1, text
    assert error_idx != -1 and exempt_idx < error_idx, (
        "source-branch exemption must be evaluated before the hard-failure error"
    )
