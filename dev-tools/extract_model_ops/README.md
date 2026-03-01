# extract\_model\_ops

Developer tool that extracts TorchScript operation sets from the supported
HuggingFace transformer architectures.  The output is used to maintain the
C++ operation allowlist in
`bin/pytorch_inference/CSupportedOperations.cc`.

## When to run

Re-run this tool whenever:

- A new transformer architecture is added to the supported set.
- The PyTorch (libtorch) version used by ml-cpp is upgraded.
- You need to verify which operations a particular model uses.

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

## Usage

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

## Configuration

The set of reference models is defined in `reference_models.json`.  Each
entry maps a short architecture name to a HuggingFace model identifier:

```json
{
    "bert": "bert-base-uncased",
    "roberta": "roberta-base"
}
```

To add a new architecture, append an entry to this file and re-run the
script.  Copy the `--cpp` output into `CSupportedOperations.cc`, adding
any new operations to the `ALLOWED_OPERATIONS` set.

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
