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
"""Reproduce the __setstate__ load-time execution gap in model validation.

TorchScript serialises a module's __setstate__ method and executes it *during*
torch::jit::load().  CModelGraphValidator, however, only inspects the (inlined)
forward graph of an already-loaded module.  A forbidden operation hidden in
__setstate__ is therefore:

  1. invisible to the validator (it never appears in the forward graph), and
  2. already executed by the time the validator would run (load happens first).

This is the timing gap raised in security review: a malicious model can read
files / crash the process at load time, before any graph check.  This test
demonstrates both facts *without needing a built binary* and only depends on
torch, so it can run in CI as a guard.

It is deliberately safe: the load-time execution proof uses a benign probe
whose __setstate__ merely print()s a marker.  The forbidden-op proofs use
static graph inspection on the saved archive and never load an attack model.

Usage:
    python3 test_setstate_load_timing.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor


# Mirror of CSupportedOperations::FORBIDDEN_OPERATIONS (subset relevant here).
FORBIDDEN_OPERATIONS = {
    "aten::from_file",
    "aten::save",
    "aten::as_strided",
}

_LOAD_MARKER = "SETSTATE_EXECUTED_DURING_LOAD"


class BenignSetStateProbe(torch.nn.Module):
    """Harmless probe: __setstate__ only print()s a marker (TorchScript-safe).

    Used to prove — without running any dangerous op — that __setstate__ code
    executes during torch.jit.load().
    """
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.zeros(1)

    @torch.jit.export
    def __getstate__(self) -> Tuple[Tensor, bool]:
        return (self.w, self.training)

    @torch.jit.export
    def __setstate__(self, state: Tuple[Tensor, bool]) -> None:
        self.w = state[0]
        self.training = state[1]
        # Literal must be inlined: TorchScript cannot close over globals.
        print("SETSTATE_EXECUTED_DURING_LOAD")

    def forward(self, x: Tensor) -> Tensor:
        return x + self.w


class SetStateFileReaderModel(torch.nn.Module):
    """aten::from_file hidden in __setstate__; forward() is benign.

    This is the attack shape: the forbidden op runs at load time and never
    appears in the forward graph the validator inspects.  We only ever inspect
    this model statically (never load it), so no file is read here.
    """
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.zeros(1)

    @torch.jit.export
    def __getstate__(self) -> Tuple[Tensor, bool]:
        return (self.w, self.training)

    @torch.jit.export
    def __setstate__(self, state: Tuple[Tensor, bool]) -> None:
        leaked = torch.from_file("/etc/passwd", size=100)
        self.w = state[0] + leaked[0]
        self.training = state[1]

    def forward(self, x: Tensor) -> Tensor:
        return x + self.w


def _forward_ops(scripted: torch.jit.ScriptModule) -> set:
    graph = scripted.forward.graph.copy()
    torch._C._jit_pass_inline(graph)
    return {node.kind() for node in graph.nodes()}


def _setstate_ops(scripted: torch.jit.ScriptModule) -> set:
    ops: set = set()
    for submodule in scripted.modules():
        try:
            graph = submodule._c._get_method("__setstate__").graph
        except (RuntimeError, AttributeError):
            continue
        ops.update(node.kind() for node in graph.nodes())
    return ops


def test_setstate_executes_during_load() -> bool:
    """Prove that __setstate__ code runs during torch.jit.load()."""
    print("--- 1. __setstate__ executes during load (benign probe) ---")
    ok = True
    with tempfile.TemporaryDirectory(prefix="setstate_timing_") as tmp:
        path = Path(tmp) / "probe.pt"
        scripted = torch.jit.script(BenignSetStateProbe())
        torch.jit.save(scripted, str(path))

        # Load in a fresh subprocess; capture what is printed *during* load.
        loader = (
            "import torch;"
            f"torch.jit.load(r'{path}');"
            "print('LOAD_RETURNED')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", loader],
            capture_output=True, text=True, timeout=120,
        )
        stdout = proc.stdout

        # The benign probe must load *cleanly*: require a zero exit and that
        # load() actually returned (LOAD_RETURNED). Otherwise a crash that
        # happened to print the marker first would be a false positive — so we
        # only accept the marker as proof when the whole load completed and the
        # marker was printed strictly before load() returned.
        loaded_cleanly = proc.returncode == 0 and "LOAD_RETURNED" in stdout
        marker_before_return = (
            loaded_cleanly
            and _LOAD_MARKER in stdout
            and stdout.index(_LOAD_MARKER) < stdout.index("LOAD_RETURNED")
        )

        if marker_before_return:
            print(f"  OK: '{_LOAD_MARKER}' printed during load — "
                  "code ran before load() returned (and before validation).")
        else:
            print("  FAIL: marker not observed during a clean load.")
            print(f"  returncode: {proc.returncode}")
            print(f"  stdout: {stdout!r}")
            if proc.stderr.strip():
                tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
                print(f"  stderr (tail):\n{tail}")
            ok = False
    print()
    return ok


def test_forbidden_op_hidden_from_forward_graph() -> bool:
    """The validator inspects only forward; the forbidden op lives in setstate."""
    print("--- 2. forbidden op is invisible to the forward-graph validator ---")
    ok = True
    scripted = torch.jit.script(SetStateFileReaderModel().eval())

    forward = _forward_ops(scripted)
    setstate = _setstate_ops(scripted)

    forward_forbidden = forward & FORBIDDEN_OPERATIONS
    setstate_forbidden = setstate & FORBIDDEN_OPERATIONS

    print(f"  forward ops    : {sorted(forward)}")
    print(f"  __setstate__ ops: {sorted(setstate)}")

    if forward_forbidden:
        print(f"  FAIL: forbidden op unexpectedly in forward: {sorted(forward_forbidden)}")
        ok = False
    else:
        print("  OK: forward graph contains no forbidden op — "
              "the current validator would ACCEPT this model.")

    if "aten::from_file" in setstate_forbidden:
        print("  OK: aten::from_file present in __setstate__ — "
              "a correct pre-load scan must catch this.")
    else:
        print("  FAIL: aten::from_file not found in __setstate__ (fixture broken).")
        ok = False
    print()
    return ok


def run_tests() -> bool:
    print("=" * 72)
    print("Repro: __setstate__ load-time execution gap (security review)")
    print("=" * 72)
    print(f"torch version: {torch.__version__}")
    print()

    results = [
        test_setstate_executes_during_load(),
        test_forbidden_op_hidden_from_forward_graph(),
    ]

    all_passed = all(results)
    print("=" * 72)
    if all_passed:
        print("ALL CHECKS PASSED — the load-time gap is reproduced and documented.")
        print("A fix must refuse archives containing __setstate__/__getstate__")
        print("BEFORE torch::jit::load() is called.")
    else:
        print("SOME CHECKS FAILED — see above.")
    print("=" * 72)
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
