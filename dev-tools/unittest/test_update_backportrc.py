#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Pytest tests for dev-tools/update_backportrc.py (minor-freeze backportrc update).

The backport tool applies the FIRST matching ``branchLabelMapping`` entry, so the
specific main-only override (``^v<main>$`` -> ``main``) must be emitted before the
generic ``^vX.Y.Z$`` -> ``X.Y`` rule. Otherwise the new main version label resolves
to a non-existent MAJOR.MINOR release branch and the backport fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEV_TOOLS = Path(__file__).resolve().parents[1]
if str(_DEV_TOOLS) not in sys.path:
    sys.path.insert(0, str(_DEV_TOOLS))

import update_backportrc as ubrc  # noqa: E402

_GENERIC_KEY = r"^v(\d+).(\d+).\d+(?:-(?:alpha|beta|rc)\d+)?$"


def _base_data() -> dict:
    return {
        "targetBranchChoices": ["main", "9.4", "9.3"],
        "branchLabelMapping": {
            _GENERIC_KEY: "$1.$2",
            "^v9.5.0$": "main",
        },
    }


def test_main_override_emitted_first() -> None:
    data = _base_data()
    changed = ubrc.update_backportrc_for_minor_freeze(
        data, new_release_branch="9.5", main_new_version="9.6.0"
    )
    assert changed is True

    keys = list(data["branchLabelMapping"].keys())
    assert keys[0] == "^v9.6.0$", f"main override must be first, got {keys}"
    assert data["branchLabelMapping"]["^v9.6.0$"] == "main"
    # The generic rule is preserved, just after the override.
    assert data["branchLabelMapping"][_GENERIC_KEY] == "$1.$2"


def test_stale_main_override_removed() -> None:
    data = _base_data()
    ubrc.update_backportrc_for_minor_freeze(
        data, new_release_branch="9.5", main_new_version="9.6.0"
    )
    # The previous main override (^v9.5.0$) must not linger.
    assert "^v9.5.0$" not in data["branchLabelMapping"]
    main_keys = [k for k, v in data["branchLabelMapping"].items() if v == "main"]
    assert main_keys == ["^v9.6.0$"]


def test_reorder_only_is_detected_as_change() -> None:
    # Mapping already has the right content but the generic rule is first.
    data = {
        "targetBranchChoices": ["main", "9.5"],
        "branchLabelMapping": {
            _GENERIC_KEY: "$1.$2",
            "^v9.6.0$": "main",
        },
    }
    changed = ubrc.update_backportrc_for_minor_freeze(
        data, new_release_branch="9.5", main_new_version="9.6.0"
    )
    assert changed is True
    assert list(data["branchLabelMapping"].keys())[0] == "^v9.6.0$"


def test_idempotent_when_already_ordered() -> None:
    data = {
        "targetBranchChoices": ["main", "9.5"],
        "branchLabelMapping": {
            "^v9.6.0$": "main",
            _GENERIC_KEY: "$1.$2",
        },
    }
    changed = ubrc.update_backportrc_for_minor_freeze(
        data, new_release_branch="9.5", main_new_version="9.6.0"
    )
    assert changed is False
    assert list(data["branchLabelMapping"].keys())[0] == "^v9.6.0$"


def test_generic_rule_preserved_exactly() -> None:
    data = _base_data()
    ubrc.update_backportrc_for_minor_freeze(
        data, new_release_branch="9.5", main_new_version="9.6.0"
    )
    assert _GENERIC_KEY in data["branchLabelMapping"]


def test_misconfigured_main_key_is_corrected() -> None:
    # Existing mapping already has the new main key but pointing at a (wrong)
    # release branch instead of "main". The override must still win.
    data = {
        "targetBranchChoices": ["main", "9.5"],
        "branchLabelMapping": {
            _GENERIC_KEY: "$1.$2",
            "^v9.6.0$": "9.6",
        },
    }
    changed = ubrc.update_backportrc_for_minor_freeze(
        data, new_release_branch="9.5", main_new_version="9.6.0"
    )
    assert changed is True
    keys = list(data["branchLabelMapping"].keys())
    assert keys[0] == "^v9.6.0$", f"main override must be first, got {keys}"
    assert data["branchLabelMapping"]["^v9.6.0$"] == "main"
