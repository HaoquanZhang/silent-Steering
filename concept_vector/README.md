# Concept Vector Extraction

This folder implements concept-vector extraction from the Transformer Circuits introspection protocol and saves vectors in a format compatible with this repo's `off-mani.sh` pipeline.

## Method (implemented)
For each word, we use:

```text
Human: Tell me about {word}

Assistant:
```

We extract residual-stream hidden states at the final prompt token (the `:` in `Assistant:`), then compute:

- baseline mean over 100 baseline words: `mu_l`
- concept vector for concept word `c`: `v_l(c) = h_l(c) - mu_l`

Outputs are saved as raw tensors in:

- `<word>_response_avg_diff.pt` (shape `[num_layers, hidden_size]`)
- optional `<word>_layerXX_response_avg_diff.pt` (shape `[hidden_size]`)

These are directly loadable by `glp/script_introspection_offmanifold.py`.

## Run

### Llama-3.1-8B
```bash
python concept_vector/extract_concept_vectors.py \
  --model meta-llama/Llama-3.1-8B \
  --output_dir concept_vector/outputs \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch_size 8
```

### Llama-3.2-1B
```bash
python concept_vector/extract_concept_vectors.py \
  --model meta-llama/Llama-3.2-1B \
  --output_dir concept_vector/outputs \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch_size 16
```

### Optional single-layer export
```bash
python concept_vector/extract_concept_vectors.py \
  --model meta-llama/Llama-3.1-8B \
  --output_dir concept_vector/outputs \
  --device cuda:0 \
  --export_layer_idx 15
```

## Quick compatibility check with `off-mani.sh`
After extraction, set:

```bash
VECTOR_PATH="concept_vector/outputs/Llama-3.1-8B/bread_response_avg_diff.pt"
LAYER_IDX=15
```

For Llama-3.2-1B, usually start with `LAYER_IDX=7`.

