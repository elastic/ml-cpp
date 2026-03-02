# extract\_model\_ops

Developer tools for maintaining and validating the TorchScript operation
allowlist in `bin/pytorch_inference/CSupportedOperations.cc`.

This directory contains two scripts that share the same Python environment:

| Script | Purpose |
|---|---|
| `extract_model_ops.py` | Generate the C++ `ALLOWED_OPERATIONS` set from reference models |
| `validate_allowlist.py` | Verify the allowlist accepts all supported models (no false positives) |

## Setup

Create a Python virtual environment and install the dependencies:

```bash
cd dev-tools/extract_model_ops
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If any of the reference models are gated, set a HuggingFace token:

```bash
export HF_TOKEN="hf_..."
```

## extract\_model\_ops.py

Traces each model in `reference_models.json`, collects the TorchScript
operations from the inlined forward graph, and outputs the union as a
sorted list or a ready-to-paste C++ initializer.

### When to run

- A new transformer architecture is added to the supported set.
- The PyTorch (libtorch) version used by ml-cpp is upgraded.
- You need to inspect which operations a particular model uses.

### Usage

```bash
# Print the sorted union of all operations (default)
python3 extract_model_ops.py

# Print a ready-to-paste C++ initializer list
python3 extract_model_ops.py --cpp

# Also show per-model breakdowns
python3 extract_model_ops.py --per-model --cpp

# Use a custom config file
python3 extract_model_ops.py --config /path/to/models.json
```

## validate\_allowlist.py

Parses `ALLOWED_OPERATIONS` and `FORBIDDEN_OPERATIONS` directly from
`CSupportedOperations.cc`, then traces every model in a config file and
checks that each model's operations are accepted.  Exits non-zero if
any model would be rejected (a false positive).

### When to run

- After regenerating `ALLOWED_OPERATIONS` with `extract_model_ops.py`.
- After adding new models to `validation_models.json`.
- As a pre-merge check for any PR that touches the allowlist or the
  graph validation logic.

### Usage

```bash
# Validate against the default set (validation_models.json)
python3 validate_allowlist.py

# Validate with verbose per-model op counts
python3 validate_allowlist.py --verbose

# Validate against a custom model set
python3 validate_allowlist.py --config /path/to/models.json
```

The script can also be run via the CMake `validate_pytorch_inference_models`
target, which automatically locates a Python 3 interpreter, creates a venv,
and installs dependencies — no manual setup required:

```bash
cmake --build cmake-build-relwithdebinfo -t validate_pytorch_inference_models
```

The CMake target searches for `python3`, `python3.12`, `python3.11`,
`python3.10`, `python3.9`, and `python` (in that order), accepting the
first one that reports Python 3.x.  This handles Linux build machines
where Python is only available as `python3.12` (via `make altinstall`)
as well as Windows where the canonical name is `python`.

## Configuration files

| File | Used by | Purpose |
|---|---|---|
| `reference_models.json` | `extract_model_ops.py` | Models whose ops form the allowlist |
| `validation_models.json` | `validate_allowlist.py` | Superset including task-specific models (NER, sentiment) from `bin/pytorch_inference/examples/` |

Each file maps a short architecture name to a HuggingFace model identifier:

```json
{
    "bert": "bert-base-uncased",
    "roberta": "roberta-base"
}
```

To add a new architecture, append an entry to `reference_models.json`,
re-run `extract_model_ops.py --cpp`, and update `CSupportedOperations.cc`.
Then add the same entry (plus any task-specific variants) to
`validation_models.json` and run `validate_allowlist.py` to confirm
there are no false positives.

## How it works

1. Each reference model is loaded via `transformers.AutoModel` with
   `torchscript=True` in the config.
2. The model is traced with `torch.jit.trace` using a short dummy input
   (falls back to `torch.jit.script` if tracing fails).
3. All method calls in the forward graph are inlined via
   `torch._C._jit_pass_inline` so that operations inside submodules
   are visible.
4. Every node's operation name (`node.kind()`) is collected, recursing
   into sub-blocks (e.g. inside `prim::If` / `prim::Loop` nodes).
5. The union across all models is reported.
