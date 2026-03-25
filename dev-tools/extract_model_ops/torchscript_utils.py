#!/usr/bin/env python3
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
"""Shared utilities for extracting and inspecting TorchScript operations."""

import importlib
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load_model_config(config_path: Path) -> dict[str, dict]:
    """Load a model config JSON file and normalise entries.

    Each entry is either a plain model-name string or a dict with
    ``model_id`` (required) and optional fields:

    - ``quantized`` (bool, default False) — apply dynamic quantization.
    - ``auto_class`` (str) — transformers Auto class name to use instead
      of ``AutoModel`` (e.g. ``"AutoModelForSequenceClassification"``).
    - ``config_overrides`` (dict) — extra kwargs passed to
      ``AutoConfig.from_pretrained`` (e.g. ``{"use_cache": false}``).

    Keys starting with ``_comment`` are silently skipped.

    Raises ``ValueError`` for malformed entries so that config problems
    are caught early with an actionable message.
    """
    with open(config_path) as f:
        raw = json.load(f)

    models: dict[str, dict] = {}
    for key, value in raw.items():
        if key.startswith("_comment"):
            continue
        if isinstance(value, str):
            models[key] = {"model_id": value, "quantized": False}
        elif isinstance(value, dict):
            if "model_id" not in value:
                raise ValueError(
                    f"Config entry {key!r} is a dict but missing required "
                    f"'model_id' key: {value!r}")
            models[key] = {
                "model_id": value["model_id"],
                "quantized": value.get("quantized", False),
                "auto_class": value.get("auto_class"),
                "config_overrides": value.get("config_overrides", {}),
            }
        else:
            raise ValueError(
                f"Config entry {key!r} has unsupported type "
                f"{type(value).__name__}: {value!r}. "
                f"Expected a model name string or a dict with 'model_id'.")
    return models


def collect_graph_ops(graph) -> set[str]:
    """Collect all operation names from a TorchScript graph, including blocks."""
    ops = set()
    for node in graph.nodes():
        ops.add(node.kind())
        for block in node.blocks():
            ops.update(collect_graph_ops(block))
    return ops


def collect_inlined_ops(module) -> set[str]:
    """Clone the forward graph, inline all calls, and return the op set."""
    graph = module.forward.graph.copy()
    torch._C._jit_pass_inline(graph)
    return collect_graph_ops(graph)


def _resolve_auto_class(class_name: str | None):
    """Resolve a transformers Auto class by name, defaulting to AutoModel."""
    if not class_name:
        return AutoModel
    import transformers
    cls = getattr(transformers, class_name, None)
    if cls is None:
        raise ValueError(f"Unknown transformers class: {class_name}")
    return cls


def load_and_trace_hf_model(model_name: str, quantize: bool = False,
                            auto_class: str | None = None,
                            config_overrides: dict | None = None):
    """Load a HuggingFace model, tokenize sample input, and trace to TorchScript.

    When *quantize* is True the model is dynamically quantized (nn.Linear
    layers converted to quantized::linear_dynamic) before tracing.  This
    mirrors what Eland does when importing models for Elasticsearch.

    *auto_class* selects a transformers Auto class by name (e.g.
    ``"AutoModelForSequenceClassification"``).  Defaults to ``AutoModel``.

    *config_overrides* supplies extra kwargs to ``AutoConfig.from_pretrained``
    (e.g. ``{"use_cache": False}`` for encoder-decoder models like BART).

    Returns the traced module, or None if the model could not be loaded or traced.
    """
    token = os.environ.get("HF_TOKEN")
    model_cls = _resolve_auto_class(auto_class)
    overrides = config_overrides or {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        config = AutoConfig.from_pretrained(
            model_name, torchscript=True, token=token, **overrides)
        model = model_cls.from_pretrained(
            model_name, config=config, token=token)
        model.eval()
    except Exception as exc:
        print(f"    LOAD ERROR: {exc}", file=sys.stderr)
        return None

    if quantize:
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8)
            print("    Applied dynamic quantization (nn.Linear -> qint8)",
                  file=sys.stderr)
        except Exception as exc:
            print(f"    QUANTIZE ERROR: {exc}", file=sys.stderr)
            return None

    inputs = tokenizer(
        "This is a sample input for graph extraction.",
        return_tensors="pt", padding="max_length",
        max_length=32, truncation=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    try:
        return torch.jit.trace(
            model, (input_ids, attention_mask), strict=False)
    except Exception as exc:
        print(f"    TRACE WARNING: {exc}", file=sys.stderr)
        print("    Falling back to torch.jit.script...", file=sys.stderr)
        try:
            return torch.jit.script(model)
        except Exception as exc2:
            print(f"    SCRIPT ERROR: {exc2}", file=sys.stderr)
            return None
