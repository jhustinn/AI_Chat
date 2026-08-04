"""
Merge QLoRA adapter dengan base model
dan simpan sebagai model merged
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Konfigurasi
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = r"E:\App\SWA\AI\finetune\qwen2.5-cs-assistant-adapter"
OUTPUT_DIR = r"E:\App\SWA\AI\models\qwen2.5-cs-assistant-merged"

print("="*50)
print("Step 1: Loading base model...")
print("="*50)

# Download/cache base model ke E: drive
cache_dir = r"E:\App\SWA\AI\models\cache"
os.makedirs(cache_dir, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
    cache_dir=cache_dir,
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL, 
    trust_remote_code=True,
    cache_dir=cache_dir,
)

print("Base model loaded!")

print("\n" + "="*50)
print("Step 2: Loading adapter...")
print("="*50)

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
print("Adapter loaded!")

print("\n" + "="*50)
print("Step 3: Merging...")
print("="*50)

merged_model = model.merge_and_unload()
print("Merge done!")

print("\n" + "="*50)
print("Step 4: Saving merged model...")
print("="*50)

os.makedirs(OUTPUT_DIR, exist_ok=True)
merged_model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

total = 0
for f in os.listdir(OUTPUT_DIR):
    fp = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(fp):
        s = os.path.getsize(fp)
        total += s
        print(f"  {f}: {s/1024/1024:.2f} MB")

print(f"\nTotal: {total/1024/1024:.2f} MB")
print(f"Saved to: {OUTPUT_DIR}")
print("\nDONE! Next: convert to GGUF")
