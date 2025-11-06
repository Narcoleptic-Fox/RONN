#!/usr/bin/env python3
"""
Export DistilBERT to ONNX format.

Requires: pip install transformers torch onnx
"""

import torch
from transformers import DistilBertModel, DistilBertTokenizer

def export_distilbert():
    model_name = "distilbert-base-uncased"

    print(f"Loading {model_name}...")
    model = DistilBertModel.from_pretrained(model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    model.eval()

    # Create dummy input
    dummy_input = tokenizer("This is a sample sentence for export", return_tensors="pt")
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        "distilbert.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=13,
    )

    print("✓ Exported distilbert.onnx")

    # Print model info
    import os
    size_mb = os.path.getsize("distilbert.onnx") / 1024 / 1024
    print(f"  Size: {size_mb:.1f} MB")

if __name__ == "__main__":
    export_distilbert()
