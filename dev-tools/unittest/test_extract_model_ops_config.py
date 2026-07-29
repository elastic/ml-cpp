#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

"""Tests for dev-tools/extract_model_ops config parsing (trust_remote_code).

The extract_model_ops tool defaults ``trust_remote_code`` to False so an
untrusted model repo cannot run arbitrary code during load (CVE-2026-5241);
it must be opt-in per model. ``torchscript_utils`` imports torch/transformers
at module load, so these tests are skipped where those deps are absent (e.g.
the lean dev-tools CI unittest env); they run in the extract_model_ops venv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="extract_model_ops requires torch")
pytest.importorskip("transformers", reason="extract_model_ops requires transformers")

_EXTRACT_DIR = Path(__file__).resolve().parents[1] / "extract_model_ops"
if str(_EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXTRACT_DIR))

import torchscript_utils as tsu  # noqa: E402


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "models.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_string_entry_defaults_trust_remote_code_false(tmp_path: Path) -> None:
    cfg = _write(tmp_path, {"bert": "bert-base-uncased"})
    models = tsu.load_model_config(cfg)
    assert models["bert"]["trust_remote_code"] is False


def test_dict_entry_defaults_trust_remote_code_false(tmp_path: Path) -> None:
    cfg = _write(tmp_path, {"m": {"model_id": "foo/bar", "quantized": True}})
    models = tsu.load_model_config(cfg)
    assert models["m"]["trust_remote_code"] is False


def test_dict_entry_opt_in_trust_remote_code(tmp_path: Path) -> None:
    cfg = _write(tmp_path, {"jina": {"model_id": "jinaai/x", "trust_remote_code": True}})
    models = tsu.load_model_config(cfg)
    assert models["jina"]["trust_remote_code"] is True


@pytest.mark.parametrize("bad_value", ["false", "true", 1, 0, None, ["true"]])
def test_non_bool_trust_remote_code_is_rejected(tmp_path: Path, bad_value) -> None:
    # A non-bool must never be silently coerced into enabling remote code:
    # e.g. the string "false" is truthy and would otherwise turn it on.
    cfg = _write(tmp_path, {"m": {"model_id": "foo/bar", "trust_remote_code": bad_value}})
    with pytest.raises(ValueError, match="trust_remote_code"):
        tsu.load_model_config(cfg)


def test_bundled_configs_only_trust_vetted_models() -> None:
    # The real configs must not silently trust arbitrary repos: only models
    # explicitly known to ship custom code may opt in.
    allowed_trusted = {"jina-embeddings-v5-text-nano"}
    for name in ("reference_models.json", "validation_models.json"):
        models = tsu.load_model_config(_EXTRACT_DIR / name)
        trusted = {k for k, v in models.items() if v["trust_remote_code"]}
        assert trusted <= allowed_trusted, f"{name} trusts unexpected models: {trusted - allowed_trusted}"
