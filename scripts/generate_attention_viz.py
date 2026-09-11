"""Generate attention heatmap visualizations for the five Task 1 example predictions.

Faculty spec §4.1 requests "5 example predictions with attention visualization (optional)".
This script loads the saved Task 1 checkpoint, runs the five held-out examples through
DistilBERT with output_attentions=True, and produces one heatmap per example under plots/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import DistilBertModel, DistilBertTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.plotting import plot_attention_heatmap


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    results_path = ROOT / "results/task1_results.json"
    checkpoint_path = ROOT / "checkpoints/task1_best.pt"
    plots_dir = ROOT / config["paths"]["plots"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        print("Task 1 results not found; run scripts/run_task1.py first.")
        return
    if not checkpoint_path.exists():
        print("Task 1 checkpoint not found; run scripts/run_task1.py first.")
        return

    result = json.loads(results_path.read_text(encoding="utf-8"))
    examples = result.get("examples", [])
    if not examples:
        print("No examples found in task1_results.json.")
        return

    model_name = config["bert"]["model_name"]
    max_length = int(config["bert"]["max_length"])
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    bert = DistilBertModel.from_pretrained(model_name, attn_implementation="eager")
    bert.eval()

    device = torch.device("cpu")  # attention viz doesn't need GPU

    for idx, example in enumerate(examples[:5], start=1):
        caption = example["caption"]
        encoded = tokenizer(
            caption,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        with torch.no_grad():
            outputs = bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        # Average attention across all heads in the last layer → (1, seq_len)
        # DistilBERT has 6 layers; take the last layer's attention
        last_layer_attention = outputs.attentions[-1]  # (1, num_heads, seq_len, seq_len)
        # Average across heads, take the CLS row (row 0) for CLS→token attention
        cls_attention = last_layer_attention[0].mean(dim=0)[0]  # (seq_len,)

        # Decode token IDs to readable tokens
        token_ids = input_ids.squeeze(0).tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        # Trim to actual non-padding tokens
        actual_length = int(attention_mask.sum().item())
        tokens = tokens[:actual_length]
        weights = cls_attention[:actual_length].numpy()

        # Normalize weights to [0, 1] for visualization
        weights = weights / (weights.max() + 1e-8)

        # Truncate to first 30 tokens for readability
        display_limit = min(30, len(tokens))
        display_tokens = tokens[:display_limit]
        display_weights = weights[:display_limit]

        true_tags = ", ".join(example.get("true_tags_in_vocabulary", []))
        title = f"Task 1 example {idx}: CLS→token attention (last layer avg)"

        output_path = plots_dir / f"task1_attention_example_{idx}.png"
        plot_attention_heatmap(display_tokens, display_weights, output_path, title)
        print(f"  Saved {output_path.name}: {caption[:80]}...")

    print(f"\nGenerated {min(5, len(examples))} attention heatmaps in {plots_dir}/")


if __name__ == "__main__":
    main()
