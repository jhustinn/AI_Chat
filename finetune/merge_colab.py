# ============================================================
# CELL 11: Merge Adapter + Convert ke GGUF (Jalankan di Colab SETELAH training)
# ============================================================
# Jalankan cell ini SETELAH training selesai dan adapter sudah disimpan

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Load base model (sudah di-cache dari training, tidak perlu download lagi)
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

# Load adapter
print("Loading adapter...")
model = PeftModel.from_pretrained(base_model, "./qwen2.5-cs-assistant-adapter")

# Merge
print("Merging adapter...")
merged_model = model.merge_and_unload()

# Save merged model
output_dir = "./qwen2.5-cs-assistant-merged"
print(f"Saving merged model to {output_dir}...")
merged_model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

# List files
for f in os.listdir(output_dir):
    size = os.path.getsize(os.path.join(output_dir, f))
    print(f"  {f}: {size/1024/1024:.2f} MB")

print("\n" + "="*50)
print("MERGE SELESAI!")
print("="*50)

# ============================================================
# CELL 12: Convert ke GGUF (4-bit quantization untuk hemat space)
# ============================================================
# Install llama.cpp converter
!pip install -q llama-cpp-python[server]

# Download convert script dari llama.cpp
!wget -q https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py

# Convert ke GGUF dengan Q4_K_M quantization
!python convert_hf_to_gguf.py \
    --outfile qwen2.5-cs-assistant-Q4_K_M.gguf \
    --outtype q4_k_m \
    {output_dir}

print("\n" + "="*50)
print("CONVERT SELESAI!")
print("="*50)

# ============================================================
# CELL 13: Download GGUF file (PENTING!)
# ============================================================
from google.colab import files

# Cek ukuran file
gguf_file = "qwen2.5-cs-assistant-Q4_K_M.gguf"
if os.path.exists(gguf_file):
    size = os.path.getsize(gguf_file)
    print(f"File GGUF: {size/1024/1024:.2f} MB")
    
    # Download
    print("Downloading...")
    files.download(gguf_file)
else:
    print("ERROR: File GGUF tidak ditemukan!")
