#!/usr/bin/env python3
"""Extract concept vectors compatible with off-mani.sh pipeline.

Protocol:
1) Prompt template: "Human: Tell me about {word}\n\nAssistant:"
2) Read residual stream hidden states at final token (the ':' in Assistant:)
3) Compute baseline mean over 100 baseline words
4) Concept vector for word c is h(c) - mean_baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from concept_vector.blog_word_lists import BASELINE_WORDS_100, CONCEPT_WORDS_50

PROMPT_TEMPLATE = "Human: Tell me about {word}\n\nAssistant:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract concept vectors from Llama models")
    p.add_argument("--model", type=str, required=True, help="HF model id or local path")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--normalize", type=str, default="none", choices=["none", "l2"])
    p.add_argument(
        "--export_layer_idx",
        type=int,
        default=None,
        help="If set, also export a single-layer [hidden] tensor for this layer index.",
    )
    p.add_argument("--max_words", type=int, default=None, help="Limit words for smoke tests")
    return p.parse_args()


def to_dtype(dtype_name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]


def prompt_for(word: str) -> str:
    return PROMPT_TEMPLATE.format(word=word.lower())


def baseline_hash() -> str:
    joined = "\n".join(BASELINE_WORDS_100)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def normalize_tensor(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "l2":
        denom = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return x / denom
    raise ValueError(mode)


def compute_hidden_stack(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    words: list[str],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Return mapping word -> tensor [num_layers, hidden_size]."""
    result: dict[str, torch.Tensor] = {}
    for i in tqdm(range(0, len(words), batch_size), desc="forward"):
        batch_words = words[i : i + batch_size]
        texts = [prompt_for(w) for w in batch_words]
        enc = tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        hidden_states = out.hidden_states[1:]
        # final prompt token index per item (should be final ':' token in Assistant:)
        token_positions = attention_mask.sum(dim=1) - 1

        for b_idx, word in enumerate(batch_words):
            pos = int(token_positions[b_idx].item())
            # Gather layer stack at position: [num_layers, hidden_size]
            layers = [h[b_idx, pos, :].float().cpu() for h in hidden_states]
            result[word] = torch.stack(layers, dim=0)
    return result


def save_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = to_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    model.to(args.device)
    model.eval()

    baseline_words = BASELINE_WORDS_100.copy()
    concept_words = CONCEPT_WORDS_50.copy()
    if args.max_words is not None:
        baseline_words = baseline_words[: args.max_words]
        concept_words = concept_words[: args.max_words]

    all_words = sorted(set(baseline_words + concept_words))
    hidden = compute_hidden_stack(model, tokenizer, all_words, batch_size=args.batch_size)

    baseline_tensor = torch.stack([hidden[w] for w in baseline_words], dim=0)  # [B, L, H]
    baseline_mean = baseline_tensor.mean(dim=0)  # [L, H]

    model_name = Path(str(args.model).rstrip("/")).name
    model_out = output_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    for word in concept_words:
        vec = hidden[word] - baseline_mean  # [L, H]
        vec = normalize_tensor(vec, args.normalize)

        tensor_path = model_out / f"{word}_response_avg_diff.pt"
        torch.save(vec, tensor_path)

        if args.export_layer_idx is not None:
            layer_vec = vec[args.export_layer_idx]
            torch.save(layer_vec, model_out / f"{word}_layer{args.export_layer_idx:02d}_response_avg_diff.pt")

        meta = {
            "model": args.model,
            "model_name": model_name,
            "word": word,
            "prompt_template": PROMPT_TEMPLATE,
            "token_position_rule": "last prompt token (final ':' in Assistant:)",
            "residual_location": "hidden_states[1:] at token index",
            "vector_formula": "v_l(c)=h_l(c)-mean_{b in baseline}(h_l(b))",
            "num_layers": int(vec.shape[0]),
            "hidden_size": int(vec.shape[1]),
            "normalize": args.normalize,
            "dtype": str(vec.dtype),
            "baseline_words_count": len(baseline_words),
            "baseline_words_sha256_16": baseline_hash(),
            "compatible_with": ["off-mani.sh", "glp/script_introspection_offmanifold.py"],
        }
        save_metadata(model_out / f"{word}_metadata.json", meta)

    print(f"Saved concept vectors to: {model_out}")
    if args.export_layer_idx is not None:
        print(f"Single-layer exports also saved for layer {args.export_layer_idx}")
    print("Example off-mani.sh setting:")
    sample_word = concept_words[0]
    print(f'  VECTOR_PATH="{model_out / (sample_word + "_response_avg_diff.pt")}"')


if __name__ == "__main__":
    main()
