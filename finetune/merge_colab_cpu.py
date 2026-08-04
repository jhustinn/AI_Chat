# ============================================================
# COMPLETE: Merge + Convert di Colab (CPU Mode)
# Paste semua kode ini ke SATU cell Colab, jalankan
# ============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Step 0: Upload adapter jika belum
if not os.path.exists("./qwen2.5-cs-assistant-adapter"):
    print("Upload adapter dulu...")
    from google.colab import files
    uploaded = files.upload()  # Upload qwen2.5-cs-assistant-adapter.zip
    !unzip -q *.zip -d .
    print("Adapter uploaded!")
else:
    print("Adapter sudah ada!")

# Step 1: Load base model
!pip install -q torchao peft --upgrade
print("\n[1/4] Loading base model (CPU)...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
print("  Done!")

# Step 2: Load adapter
print("\n[2/4] Loading adapter...")
model = PeftModel.from_pretrained(model, "./qwen2.5-cs-assistant-adapter")
print("  Done!")

# Step 3: Merge
print("\n[3/4] Merging...")
merged = model.merge_and_unload()
print("  Done!")

# Step 4: Save
print("\n[4/4] Saving merged model...")
output_dir = "./qwen2.5-cs-assistant-merged"
merged.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
for f in os.listdir(output_dir):
    s = os.path.getsize(os.path.join(output_dir, f))
    print(f"  {f}: {s/1024/1024:.1f} MB")

# Step 5: Convert ke GGUF
print("\n[5/5] Converting ke GGUF...")
!pip install -q llama-cpp-python
!git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp 2>/dev/null || true
!python /tmp/llama.cpp/convert_hf_to_gguf.py \
    --outfile qwen2.5-cs-assistant.gguf \
    --outtype q4_k_m \
    ./qwen2.5-cs-assistant-merged

# Step 6: Download
from google.colab import files
gguf = "qwen2.5-cs-assistant.gguf"
if os.path.exists(gguf):
    print(f"\nGGUF: {os.path.getsize(gguf)/1024/1024:.1f} MB")
    print("Downloading ke laptop...")
    files.download(gguf)
else:
    print("ERROR: GGUF tidak dibuat!")
