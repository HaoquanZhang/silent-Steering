# Concept Vector Reimplementation Plan (Llama-3.1-8B + Llama-3.2-1B)

## Goal
Reimplement the Transformer Circuits introspection blog's concept-vector extraction protocol for:
- `meta-llama/Llama-3.1-8B`
- `meta-llama/Llama-3.2-1B`

and store vectors in a format directly usable by this repo's `off-mani.sh` / `glp/script_introspection_offmanifold.py` pipeline.

---

## Blog-accurate Concept Vector Extraction Details

### 1) Prompt used to extract concept activations
For each target word `{word}` (always lowercase), use the transcript:

```text
Human: Tell me about {word}

Assistant:
```

### 2) Exact token position to read from
Extract activations from the **final `:` token** in `Assistant:` (i.e., the token right before generation begins).

Implementation note:
- We will tokenize the full prompt and programmatically identify the final token position corresponding to the trailing colon in `Assistant:`.
- If tokenization splits punctuation unexpectedly, we will match the **last token index of the prompt** after confirming it corresponds to the assistant-prefix terminal token in decoded text.

### 3) Layer/location of activation
Use the **residual stream hidden state** at that token position.

### 4) Baseline average hidden state
The blog computes a baseline by averaging activations over a fixed list of **100 baseline words** using the same prompt/template and same token position.

Let:
- `h_l(w)` = residual activation at layer `l` on the final `:` token for prompt with word `w`.
- `B` = set of 100 baseline words.

Then baseline mean at layer `l` is:

```text
mu_l = (1 / |B|) * sum_{b in B} h_l(b)
```

### 5) Concept vector for each target word
For each target concept word `c` (from the 50-word concept list), compute:

```text
v_l(c) = h_l(c) - mu_l
```

This is **not** a pairwise contrast (`pos - neg`) between two prompts; it is a **single positive prompt minus the many-word baseline mean**.

### 6) Which words to use
- Reproduce the blog's primary set of **50 concept words** (main experiments).
- Also support optional extraction for the additional control-word list used in the intentional-control appendix.
- Keep word case lowercased when inserted into the extraction template.

### 7) Injection compatibility context
The blog injects vectors into the residual stream and, in main introspection prompts, begins injection at the double-newline token before `Trial 1` and continues through later tokens. For this repo, extraction output only needs to be compatible with downstream steering in `off-mani.sh`.

---

## Compatibility Requirements (for `off-mani.sh`)
The off-manifold pipeline loads vectors with `torch.load(...)` and accepts either:
1. A raw tensor (`torch.Tensor`), or
2. A dict containing a tensor under one of these keys:
   - `vector`
   - `steering_vector`
   - `response_avg_diff`
   - `avg_diff`

To maximize compatibility, we will save:
- a primary raw tensor file: `<concept_name>_response_avg_diff.pt`
- and a metadata sidecar: `<concept_name>_metadata.json`

Recommended tensor shape:
- `[hidden_size]` for single-layer export, or
- `[num_layers, hidden_size]` for all-layer export,
so `script_introspection_offmanifold.py` can select by `--layer_idx` without modifications.

---

## Proposed Folder Layout
```text
concept_vector/
  PLAN.md
  README.md
  extract_concept_vectors.py
  blog_word_lists.py
  configs/
    llama31_8b.yaml
    llama32_1b.yaml
  outputs/
    Llama-3.1-8B/
      <concept>_response_avg_diff.pt
      <concept>_metadata.json
    Llama-3.2-1B/
      <concept>_response_avg_diff.pt
      <concept>_metadata.json
```

---

## Implementation Plan
1. **Encode canonical word lists from the blog**
   - Add the 100 baseline words and 50 concept words as constants.
   - Add optional control-word list for appendix replication.

2. **Build extraction prompt + token-position resolver**
   - Format exactly: `Human: Tell me about {word}\n\nAssistant:`.
   - Resolve the final `Assistant:` colon token index robustly.

3. **Extract residual activations**
   - Run forward pass with `output_hidden_states=True`.
   - Collect hidden states at the chosen token index for every layer.
   - Cache `h_l(w)` per word.

4. **Compute baseline mean and concept vectors**
   - Compute `mu_l` by averaging over the 100 baseline words.
   - For each concept word `c`, compute `v_l(c) = h_l(c) - mu_l`.
   - Optional flags: `--normalize none|l2` (default `none` for faithful reproduction).

5. **Serialize in off-mani-compatible format**
   - Save per-concept tensor as `.pt` (raw tensor).
   - Save metadata JSON including: model id, prompt template, token index rule, layer count, hidden size, baseline word hash, concept word, dtype.

6. **Validation checks**
   - Dimension checks against model hidden size.
   - Determinism check across seeds (for extraction there should be no sampling variance).
   - Print ready-to-use `off-mani.sh` parameter snippet (`VECTOR_PATH`, suggested `LAYER_IDX`).

7. **Runbooks for both models**
   - Provide exact commands for Llama-3.1-8B and Llama-3.2-1B.
   - Include default GLP layer recommendations aligned with this repo:
     - 8B: layer 15
     - 1B: layer 7

---

## Deliverables
- `concept_vector/extract_concept_vectors.py`
- `concept_vector/blog_word_lists.py`
- `concept_vector/README.md` with end-to-end commands
- `concept_vector/configs/llama31_8b.yaml`
- `concept_vector/configs/llama32_1b.yaml`
- output `.pt` artifacts in `concept_vector/outputs/...` compatible with `off-mani.sh`

