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
"""Extract TorchScript operation sets from supported HuggingFace transformer architectures.

This developer tool traces/scripts reference models and collects the set of
TorchScript operations that appear in their forward() computation graphs.
The output is a sorted, de-duplicated union of all operations which can be
used to build the C++ allowlist in CSupportedOperations.h.

Usage:
    python3 extract_model_ops.py [--per-model] [--cpp] [--config CONFIG]

Flags:
    --per-model      Print the op set for each model individually.
    --cpp            Print the union as a C++ initializer list.
    --config CONFIG  Path to the reference models JSON config file.
                     Defaults to reference_models.json in the same directory.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "reference_models.json"


def load_reference_models(config_path: Path) -> dict[str, str]:
    """Load the architecture-to-model mapping from a JSON config file."""
    with open(config_path) as f:
        return json.load(f)


def collect_graph_ops(graph):
    """Collect all operation names from a TorchScript graph, including blocks."""
    ops = set()
    for node in graph.nodes():
        ops.add(node.kind())
        for block in node.blocks():
            ops.update(collect_graph_ops(block))
    return ops


def collect_all_module_ops(module):
    """Collect all ops by inlining method calls and walking the flattened graph."""
    forward = module.forward
    graph = forward.graph.copy()
    torch._C._jit_pass_inline(graph)
    return collect_graph_ops(graph)


def extract_ops_for_model(model_name: str) -> set[str]:
    """Trace a HuggingFace model and return its TorchScript op set."""
    print(f"  Loading {model_name}...", file=sys.stderr)
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    config = AutoConfig.from_pretrained(model_name, torchscript=True, token=token)
    model = AutoModel.from_pretrained(model_name, config=config, token=token)
    model.eval()

    inputs = tokenizer("This is a sample input for graph extraction.",
                       return_tensors="pt", padding="max_length",
                       max_length=32, truncation=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    try:
        traced = torch.jit.trace(model, (input_ids, attention_mask), strict=False)
    except Exception as e:
        print(f"  Warning: trace failed for {model_name}: {e}", file=sys.stderr)
        print(f"  Falling back to torch.jit.script...", file=sys.stderr)
        try:
            traced = torch.jit.script(model)
        except Exception as e2:
            print(f"  Error: script also failed for {model_name}: {e2}", file=sys.stderr)
            return set()

    return collect_all_module_ops(traced)


def format_cpp_initializer(ops: set[str]) -> str:
    """Format the op set as a C++ initializer list for std::unordered_set."""
    sorted_ops = sorted(ops)
    lines = []
    for op in sorted_ops:
        lines.append(f'    "{op}"sv,')
    return "{\n" + "\n".join(lines) + "\n}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-model", action="store_true",
                        help="Print per-model op sets")
    parser.add_argument("--cpp", action="store_true",
                        help="Print union as C++ initializer")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to reference_models.json config file")
    args = parser.parse_args()

    reference_models = load_reference_models(args.config)

    per_model_ops = {}
    union_ops = set()

    print("Extracting TorchScript ops from supported architectures...",
          file=sys.stderr)

    for arch, model_name in reference_models.items():
        ops = extract_ops_for_model(model_name)
        per_model_ops[arch] = ops
        union_ops.update(ops)
        print(f"  {arch}: {len(ops)} ops", file=sys.stderr)

    print(f"\nTotal union: {len(union_ops)} unique ops", file=sys.stderr)

    if args.per_model:
        for arch, ops in sorted(per_model_ops.items()):
            print(f"\n=== {arch} ({reference_models[arch]}) ===")
            for op in sorted(ops):
                print(f"  {op}")

    if args.cpp:
        print("\n// C++ initializer for SUPPORTED_OPERATIONS:")
        print(format_cpp_initializer(union_ops))
    else:
        print("\n// Sorted union of all operations:")
        for op in sorted(union_ops):
            print(op)


if __name__ == "__main__":
    main()
