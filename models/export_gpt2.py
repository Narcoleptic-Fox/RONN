#!/usr/bin/env python3
"""
Export GPT-2 Small to ONNX format.

Requires: pip install transformers torch onnx
"""

import torch
from transformers import GPT2Model, GPT2Tokenizer

def export_gpt2():
    model_name = "gpt2"  # This is GPT-2 Small (124M params)

    print(f"Loading {model_name}...")
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    model.eval()

    # Create dummy input
    dummy_text = "This is a sample text for GPT-2 export"
    dummy_input = tokenizer(dummy_text, return_tensors="pt")
    input_ids = dummy_input["input_ids"]

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        input_ids,
        "gpt2-small.onnx",
        input_names=["input_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=13,
    )

    print("✓ Exported gpt2-small.onnx")

    # Print model info
    import os
    size_mb = os.path.getsize("gpt2-small.onnx") / 1024 / 1024
    print(f"  Size: {size_mb:.1f} MB")

if __name__ == "__main__":
    export_gpt2()
