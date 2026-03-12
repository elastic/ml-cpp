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

import os
import sys

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


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


def load_and_trace_hf_model(model_name: str):
    """Load a HuggingFace model, tokenize sample input, and trace to TorchScript.

    Returns the traced module, or None if the model could not be loaded or traced.
    """
    token = os.environ.get("HF_TOKEN")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        config = AutoConfig.from_pretrained(
            model_name, torchscript=True, token=token)
        model = AutoModel.from_pretrained(
            model_name, config=config, token=token)
        model.eval()
    except Exception as exc:
        print(f"    LOAD ERROR: {exc}", file=sys.stderr)
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
