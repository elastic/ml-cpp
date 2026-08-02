# Elasticsearch Integration Test Models

Pre-saved TorchScript `.pt` files extracted from the base64-encoded models
in the Elasticsearch Java integration tests. These are tiny synthetic models
(not real transformer architectures) used to test the `pytorch_inference`
loading and evaluation pipeline.

| File | Source | Description |
|------|--------|-------------|
| `supersimple_pytorch_model_it.pt` | `PyTorchModelIT.java` | Returns `torch.ones` of shape `(batch, 2)` |
| `tiny_text_expansion.pt` | `TextExpansionQueryIT.java` | Sparse weight vector sized by max input ID |
| `tiny_text_embedding.pt` | `TextEmbeddingQueryIT.java` | Random 100-dim embedding seeded by input hash |

## Regenerating

If the Java test models change, re-extract them by running the generation
snippet from this repository's root:

```bash
python3 -c "
import re, base64, os

JAVA_DIR = '<path-to-elasticsearch>/x-pack/plugin/ml/qa/native-multi-node-tests/src/javaRestTest/java/org/elasticsearch/xpack/ml/integration'
OUTPUT_DIR = 'dev-tools/extract_model_ops/es_it_models'

SOURCES = {
    'supersimple_pytorch_model_it.pt': ('PyTorchModelIT.java', 'BASE_64_ENCODED_MODEL'),
    'tiny_text_expansion.pt': ('TextExpansionQueryIT.java', 'BASE_64_ENCODED_MODEL'),
    'tiny_text_embedding.pt': ('TextEmbeddingQueryIT.java', 'BASE_64_ENCODED_MODEL'),
}
os.makedirs(OUTPUT_DIR, exist_ok=True)
for out_name, (java_file, var_name) in SOURCES.items():
    with open(os.path.join(JAVA_DIR, java_file)) as f:
        src = f.read()
    m = re.search(rf'{var_name}\s*=\s*(\".*?\");', src, re.DOTALL)
    b64 = re.sub(r'\"\s*\+\s*\"', '', m.group(1)).strip('\"').replace('\n', '').replace(' ', '')
    with open(os.path.join(OUTPUT_DIR, out_name), 'wb') as f:
        f.write(base64.b64decode(b64))
    print(f'Wrote {out_name}')
"
```
