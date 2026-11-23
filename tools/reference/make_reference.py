import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = Path("models/tinyllama-15M")
OUT_DIR = Path("engine/src/test/resources/reference")
PROMPT = "Once upon a time there was a small dragon."

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float32,
).eval()

inputs = tokenizer(PROMPT, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits.cpu().numpy().astype(np.float32)

logits.tofile(OUT_DIR / "reference_logits.bin")

with open(OUT_DIR / "reference_tokens.json", "w") as f:
    json.dump(inputs["input_ids"][0].tolist(), f)

metadata = {
    "prompt": PROMPT,
    "shape": list(logits.shape),
    "dtype": "float32",
    "torch": torch.__version__,
    "transformers": transformers.__version__,
}

with open(OUT_DIR / "reference_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Done.")
print("Shape:", logits.shape)