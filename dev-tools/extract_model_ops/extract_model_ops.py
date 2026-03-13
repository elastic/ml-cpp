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
    python3 extract_model_ops.py [--per-model] [--cpp] [--golden OUTPUT] [--config CONFIG]

Flags:
    --per-model      Print the op set for each model individually.
    --cpp            Print the union as a C++ initializer list.
    --golden OUTPUT  Write per-model op sets as a JSON golden file for the
                     C++ allowlist drift test.
    --config CONFIG  Path to the reference models JSON config file.
                     Defaults to reference_models.json in the same directory.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from torchscript_utils import collect_inlined_ops, load_and_trace_hf_model

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "reference_models.json"


def load_reference_models(config_path: Path) -> dict[str, str]:
    """Load the architecture-to-model mapping from a JSON config file."""
    with open(config_path) as f:
        return json.load(f)


def extract_ops_for_model(model_name: str) -> set[str] | None:
    """Trace a HuggingFace model and return its TorchScript op set.

    Returns None if the model could not be loaded or traced.
    """
    print(f"  Loading {model_name}...", file=sys.stderr)
    traced = load_and_trace_hf_model(model_name)
    if traced is None:
        return None
    return collect_inlined_ops(traced)


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
    parser.add_argument("--golden", type=Path, default=None, metavar="OUTPUT",
                        help="Write per-model op sets as a JSON golden file")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to reference_models.json config file")
    args = parser.parse_args()

    reference_models = load_reference_models(args.config)

    per_model_ops = {}
    union_ops = set()

    print("Extracting TorchScript ops from supported architectures...",
          file=sys.stderr)

    failed = []
    for arch, model_name in reference_models.items():
        ops = extract_ops_for_model(model_name)
        if ops is None:
            failed.append(arch)
            print(f"  {arch}: FAILED", file=sys.stderr)
            continue
        per_model_ops[arch] = ops
        union_ops.update(ops)
        print(f"  {arch}: {len(ops)} ops", file=sys.stderr)

    print(f"\nTotal union: {len(union_ops)} unique ops", file=sys.stderr)
    if failed:
        print(f"Failed models: {', '.join(failed)}", file=sys.stderr)

    if args.golden:
        golden = {
            "pytorch_version": torch.__version__,
            "models": {
                arch: {
                    "model_id": reference_models[arch],
                    "ops": sorted(ops),
                }
                for arch, ops in sorted(per_model_ops.items())
            },
        }
        args.golden.parent.mkdir(parents=True, exist_ok=True)
        with open(args.golden, "w") as f:
            json.dump(golden, f, indent=2)
            f.write("\n")
        print(f"Wrote golden file to {args.golden} "
              f"({len(per_model_ops)} models, "
              f"{len(union_ops)} unique ops)", file=sys.stderr)

    if args.per_model:
        for arch, ops in sorted(per_model_ops.items()):
            print(f"\n=== {arch} ({reference_models[arch]}) ===")
            for op in sorted(ops):
                print(f"  {op}")

    if args.cpp:
        print("\n// C++ initializer for SUPPORTED_OPERATIONS:")
        print(format_cpp_initializer(union_ops))
    elif not args.golden:
        print("\n// Sorted union of all operations:")
        for op in sorted(union_ops):
            print(op)


if __name__ == "__main__":
    main()
