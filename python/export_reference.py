import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "../models/tinyllama-15M"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,
)

prompt = "Hello"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits[0, -1].tolist()

result = {
    "prompt": prompt,
    "input_ids": inputs["input_ids"][0].tolist(),
    "logits": logits,
}

with open("reference.json", "w") as f:
    json.dump(result, f)

print("Input IDs:", result["input_ids"])
print("Vocabulary size:", len(logits))
print("Reference written to reference.json")