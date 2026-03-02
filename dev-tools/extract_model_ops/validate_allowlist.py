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
"""Validate that the C++ operation allowlist accepts all supported model architectures.

Traces each model listed in a JSON config file, extracts its TorchScript
operations (using the same inlining approach as the C++ validator), and
checks every operation against the ALLOWED_OPERATIONS and FORBIDDEN_OPERATIONS
sets parsed from CSupportedOperations.cc.

This is the Python-side equivalent of the C++ CModelGraphValidator and is
intended as an integration test: if any legitimate model produces an
operation that the C++ code would reject, this script exits non-zero.

Exit codes:
    0  All models pass (no false positives).
    1  At least one model was rejected or a model failed to load/trace.

Usage:
    python3 validate_allowlist.py [--config CONFIG] [--verbose]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = SCRIPT_DIR / "validation_models.json"
SUPPORTED_OPS_CC = REPO_ROOT / "bin" / "pytorch_inference" / "CSupportedOperations.cc"


def parse_string_set_from_cc(path: Path, variable_name: str) -> set[str]:
    """Extract a set of string literals from a C++ TStringViewSet definition."""
    text = path.read_text()
    pattern = rf'{re.escape(variable_name)}\s*=\s*\{{(.*?)\}};'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find {variable_name} in {path}")
    block = match.group(1)
    return set(re.findall(r'"([^"]+)"', block))


def load_cpp_sets() -> tuple[set[str], set[str]]:
    """Parse ALLOWED_OPERATIONS and FORBIDDEN_OPERATIONS from the C++ source."""
    allowed = parse_string_set_from_cc(SUPPORTED_OPS_CC, "ALLOWED_OPERATIONS")
    forbidden = parse_string_set_from_cc(SUPPORTED_OPS_CC, "FORBIDDEN_OPERATIONS")
    return allowed, forbidden


def collect_graph_ops(graph) -> set[str]:
    """Collect all operation names from a TorchScript graph, including blocks."""
    ops = set()
    for node in graph.nodes():
        ops.add(node.kind())
        for block in node.blocks():
            ops.update(collect_graph_ops(block))
    return ops


def trace_and_collect_ops(model_name: str) -> set[str] | None:
    """Load, trace, inline, and return the op set for a HuggingFace model.

    Returns None if the model could not be loaded or traced.
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
        traced = torch.jit.trace(
            model, (input_ids, attention_mask), strict=False)
    except Exception as exc:
        print(f"    TRACE WARNING: {exc}", file=sys.stderr)
        print("    Falling back to torch.jit.script...", file=sys.stderr)
        try:
            traced = torch.jit.script(model)
        except Exception as exc2:
            print(f"    SCRIPT ERROR: {exc2}", file=sys.stderr)
            return None

    graph = traced.forward.graph.copy()
    torch._C._jit_pass_inline(graph)
    return collect_graph_ops(graph)


def load_pt_and_collect_ops(pt_path: str) -> set[str] | None:
    """Load a saved TorchScript .pt file, inline, and return its op set."""
    try:
        module = torch.jit.load(pt_path)
        graph = module.forward.graph.copy()
        torch._C._jit_pass_inline(graph)
        return collect_graph_ops(graph)
    except Exception as exc:
        print(f"    LOAD ERROR: {exc}", file=sys.stderr)
        return None


def check_ops(ops: set[str],
              allowed: set[str],
              forbidden: set[str],
              verbose: bool) -> bool:
    """Check an op set against allowed/forbidden lists. Returns True if all pass."""
    forbidden_found = sorted(ops & forbidden)
    unrecognised = sorted(ops - allowed - forbidden)

    if verbose:
        print(f"    {len(ops)} distinct ops", file=sys.stderr)

    if forbidden_found:
        print(f"    FORBIDDEN: {forbidden_found}", file=sys.stderr)
    if unrecognised:
        print(f"    UNRECOGNISED: {unrecognised}", file=sys.stderr)

    if not forbidden_found and not unrecognised:
        print(f"    PASS", file=sys.stderr)
        return True

    print(f"    FAIL", file=sys.stderr)
    return False


def validate_model(model_name: str,
                   allowed: set[str],
                   forbidden: set[str],
                   verbose: bool) -> bool:
    """Validate one HuggingFace model. Returns True if all ops pass."""
    print(f"  {model_name}...", file=sys.stderr)
    ops = trace_and_collect_ops(model_name)
    if ops is None:
        print(f"    FAILED (could not load/trace)", file=sys.stderr)
        return False
    return check_ops(ops, allowed, forbidden, verbose)


def validate_pt_file(name: str,
                     pt_path: str,
                     allowed: set[str],
                     forbidden: set[str],
                     verbose: bool) -> bool:
    """Validate a local TorchScript .pt file. Returns True if all ops pass."""
    print(f"  {name} ({pt_path})...", file=sys.stderr)
    ops = load_pt_and_collect_ops(pt_path)
    if ops is None:
        print(f"    FAILED (could not load)", file=sys.stderr)
        return False
    return check_ops(ops, allowed, forbidden, verbose)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Path to reference_models.json (default: %(default)s)")
    parser.add_argument(
        "--pt-dir", type=Path, default=None,
        help="Directory of pre-saved .pt TorchScript files to validate")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-model op counts")
    args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}", file=sys.stderr)

    allowed, forbidden = load_cpp_sets()
    print(f"Parsed {len(allowed)} allowed ops and {len(forbidden)} "
          f"forbidden ops from {SUPPORTED_OPS_CC.name}", file=sys.stderr)

    results: dict[str, bool] = {}

    with open(args.config) as f:
        models = json.load(f)
    print(f"Validating {len(models)} HuggingFace models from "
          f"{args.config.name}...", file=sys.stderr)

    for arch, model_id in models.items():
        results[arch] = validate_model(
            model_id, allowed, forbidden, args.verbose)

    if args.pt_dir and args.pt_dir.is_dir():
        pt_files = sorted(args.pt_dir.glob("*.pt"))
        if pt_files:
            print(f"Validating {len(pt_files)} local .pt files from "
                  f"{args.pt_dir}...", file=sys.stderr)
            for pt_path in pt_files:
                name = pt_path.stem
                results[f"pt:{name}"] = validate_pt_file(
                    name, str(pt_path), allowed, forbidden, args.verbose)

    print(file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    all_pass = all(results.values())
    for key, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if key.startswith("pt:"):
            print(f"  {key}: {status}", file=sys.stderr)
        else:
            print(f"  {key} ({models[key]}): {status}", file=sys.stderr)

    print("=" * 60, file=sys.stderr)
    if all_pass:
        print("All models PASS - no false positives.", file=sys.stderr)
    else:
        failed = [a for a, p in results.items() if not p]
        print(f"FAILED models: {', '.join(failed)}", file=sys.stderr)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
