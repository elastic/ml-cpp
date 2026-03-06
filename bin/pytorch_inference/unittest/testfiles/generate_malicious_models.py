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
from typing import Optional


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
    """Hides aten::sin (unrecognised) three levels deep in submodules."""
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
        return torch.sin(x)


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


# --- Generation logic ---


MODELS = {
    "malicious_file_reader.pt": FileReaderModel,
    "malicious_mixed_file_reader.pt": MixedFileReaderModel,
    "malicious_hidden_in_submodule.pt": HiddenInSubmodule,
    "malicious_conditional.pt": ConditionalMalicious,
    "malicious_many_unrecognised.pt": ManyUnrecognisedOps,
    "malicious_file_reader_in_submodule.pt": FileReaderInSubmodule,
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
            print(f"    ops: {ops}")

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
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "malicious_models"
    print(f"Generating malicious model fixtures in {out_dir}")
    success = generate(out_dir)
    sys.exit(0 if success else 1)
