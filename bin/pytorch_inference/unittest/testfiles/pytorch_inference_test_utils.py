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
"""Shared utilities for pytorch_inference integration tests.

This module provides reusable helpers for:

  * TorchScript model compilation and serialisation
  * Binary framing in the CBufferedIStreamAdapter format (the 4-byte
    big-endian size header that Elasticsearch uses to send models to the
    pytorch_inference process)
  * Auto-discovery of the pytorch_inference binary across standard
    build directory layouts (CMake, Gradle)
  * Running pytorch_inference as a subprocess with proper arguments

Typical usage from another test script:

    from pytorch_inference_test_utils import (
        script_and_save_model,
        prepare_restore_file,
        find_pytorch_inference,
        run_pytorch_inference,
    )

    # Save a TorchScript model
    script_and_save_model(MyModel(), Path("/tmp/my_model.pt"))

    # Wrap it in the binary framing format and run the binary
    binary = find_pytorch_inference()
    exit_code, stdout, stderr = run_pytorch_inference(
        binary, Path("/tmp/my_model.pt"), tmp_dir
    )
"""

import os
import platform
import struct
import subprocess
from pathlib import Path
from typing import Optional, Union

import torch


# ---------------------------------------------------------------------------
# Model compilation and serialisation
# ---------------------------------------------------------------------------


def script_and_save_model(
    model: torch.nn.Module,
    output_path: Union[str, Path],
    *,
    eval_mode: bool = True,
) -> Path:
    """TorchScript-compile a model and save it as a .pt archive.

    Args:
        model:       An nn.Module instance to compile via torch.jit.script.
        output_path: Destination file path for the saved .pt archive.
        eval_mode:   If True (default), call model.eval() before scripting.
                     Disabling dropout and similar layers matches inference
                     behaviour.

    Returns:
        The resolved Path of the saved file.
    """
    output_path = Path(output_path)
    if eval_mode:
        model.eval()
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# CBufferedIStreamAdapter binary framing
# ---------------------------------------------------------------------------


def prepare_restore_file(
    model_path: Union[str, Path],
    restore_path: Union[str, Path],
) -> Path:
    """Wrap a .pt archive with the size-prefixed binary framing that
    pytorch_inference expects.

    The pytorch_inference process reads models through
    CBufferedIStreamAdapter, which expects:

        [4 bytes: uint32 network-byte-order (big-endian) model size]
        [N bytes: raw model archive]

    This matches the framing that Elasticsearch writes when it sends a
    model over the named-pipe / stdin transport to the native process.

    Args:
        model_path:   Path to the raw .pt archive produced by torch.jit.save.
        restore_path: Destination path for the size-prefixed binary file.

    Returns:
        The resolved Path of the restore file.
    """
    model_path = Path(model_path)
    restore_path = Path(restore_path)

    model_bytes = model_path.read_bytes()
    with open(restore_path, "wb") as f:
        f.write(struct.pack("!I", len(model_bytes)))
        f.write(model_bytes)
    return restore_path


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

_CMAKE_BUILD_DIR_NAMES = [
    "cmake-build-relwithdebinfo",
    "cmake-build-debug",
    "cmake-build-release",
]


def find_pytorch_inference(
    project_root: Optional[Union[str, Path]] = None,
) -> str:
    """Locate the pytorch_inference binary in standard build locations.

    Searches, in order:
      1. macOS Gradle distribution bundle
      2. Linux Gradle distribution bundle
      3. CMake build directories (RelWithDebInfo, Debug, Release)

    Args:
        project_root: Explicit path to the ml-cpp repository root.  If None,
                      inferred from this file's location (assumes this module
                      lives at bin/pytorch_inference/unittest/testfiles/).

    Returns:
        Absolute path to the pytorch_inference executable.

    Raises:
        FileNotFoundError: if no executable is found in any candidate location.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    else:
        project_root = Path(project_root).resolve()

    machine = platform.machine()
    if machine in ("arm64", "aarch64"):
        darwin_arch = "darwin-aarch64"
        linux_arch = "linux-aarch64"
    else:
        darwin_arch = "darwin-x86_64"
        linux_arch = "linux-x86_64"

    candidates = [
        # macOS Gradle distribution bundle
        project_root / "build" / "distribution" / "platform" / darwin_arch
        / "controller.app" / "Contents" / "MacOS" / "pytorch_inference",
        # Linux Gradle distribution
        project_root / "build" / "distribution" / "platform" / linux_arch
        / "bin" / "pytorch_inference",
    ]

    for build_dir in _CMAKE_BUILD_DIR_NAMES:
        candidates.append(
            project_root / build_dir / "bin" / "pytorch_inference" / "pytorch_inference"
        )

    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)

    raise FileNotFoundError(
        "Could not find pytorch_inference binary.  Build the project first, "
        "or pass an explicit binary path."
    )


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


def run_pytorch_inference(
    binary: Union[str, Path],
    model_path: Union[str, Path],
    tmp_dir: Union[str, Path],
    *,
    timeout: int = 30,
    extra_args: Optional[list[str]] = None,
) -> tuple[int, str, str]:
    """Run pytorch_inference against a model file.

    Wraps the .pt archive in the CBufferedIStreamAdapter framing format,
    then invokes the binary as a subprocess.

    Args:
        binary:     Path to the pytorch_inference executable.
        model_path: Path to the .pt model archive.
        tmp_dir:    Temporary directory for the size-prefixed restore file.
        timeout:    Maximum seconds to wait for the process (default 30).
        extra_args: Additional command-line arguments to pass to the binary.

    Returns:
        Tuple of (exit_code, stdout, stderr) where stdout and stderr are
        decoded as UTF-8.

    Raises:
        subprocess.TimeoutExpired: if the process exceeds the timeout.
    """
    model_path = Path(model_path)
    tmp_dir = Path(tmp_dir)

    restore_file = tmp_dir / f"{model_path.stem}_restore.bin"
    prepare_restore_file(model_path, restore_file)

    cmd = [
        str(binary),
        f"--restore={restore_file}",
        "--validElasticLicenseKeyConfirmed=true",
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return (
        proc.returncode,
        proc.stdout.decode("utf-8", errors="replace"),
        proc.stderr.decode("utf-8", errors="replace"),
    )
