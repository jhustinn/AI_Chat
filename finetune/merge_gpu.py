"""
Merge QLoRA adapter - pakai GPU GTX 1650
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = r"E:\App\SWA\AI\finetune\qwen2.5-cs-assistant-adapter"
OUTPUT_DIR = r"E:\App\SWA\AI\models\qwen2.5-cs-assistant-merged"
CACHE_DIR = r"E:\App\SWA\AI\models\cache"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Cek GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: GPU tidak terdeteksi, pakai CPU")

print("\n[1/4] Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",  # Otomatis pakai GPU jika tersedia
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, cache_dir=CACHE_DIR)
print("  Done!")

print("\n[2/4] Loading adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
print("  Done!")

print("\n[3/4] Merging...")
merged = model.merge_and_unload()
print("  Done!")

print("\n[4/4] Saving...")
merged.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

total = 0
for f in os.listdir(OUTPUT_DIR):
    fp = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(fp):
        s = os.path.getsize(fp)
        total += s
        print(f"  {f}: {s/1024/1024:.1f} MB")

print(f"\nTotal: {total/1024/1024:.1f} MB")
print(f"Saved: {OUTPUT_DIR}")
print("\nDONE!")
