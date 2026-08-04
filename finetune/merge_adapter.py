"""
Script untuk merge LoRA adapter dengan base model
dan convert ke format GGUF untuk llama.cpp server

Jalankan di lokal setelah download adapter dari Colab
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil

# ============================================================
# Konfigurasi - SESUAIKAN PATH INI
# ============================================================
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Atau path lokal jika sudah download
ADAPTER_PATH = "./qwen2.5-cs-assistant-adapter"  # Path adapter dari Colab
OUTPUT_DIR = "./qwen2.5-cs-assistant-merged"  # Output merged model
GGUF_OUTPUT = "./qwen2.5-cs-assistant.gguf"  # Output GGUF untuk llama.cpp

# ============================================================
# Step 1: Load Base Model (tanpa quantization)
# ============================================================
print("="*50)
print("Step 1: Loading base model...")
print("="*50)

# Cek apakah base model sudah ada di lokal
local_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "qwen2.5-1.5b")
if os.path.exists(local_model_path):
    print(f"Loading dari lokal: {local_model_path}")
    base_model_path = local_model_path
else:
    print(f"Downloading dari HuggingFace: {BASE_MODEL}")
    print("(Ini akan download ~3GB, tunggu sebentar...)")
    base_model_path = BASE_MODEL

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cpu",  # Pakai CPU untuk merge (lebih aman)
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

print("Base model loaded!")

# ============================================================
# Step 2: Load dan Merge LoRA Adapter
# ============================================================
print("\n" + "="*50)
print("Step 2: Loading adapter...")
print("="*50)

if not os.path.exists(ADAPTER_PATH):
    print(f"ERROR: Adapter tidak ditemukan di {ADAPTER_PATH}")
    print("Pastikan sudah download adapter dari Colab dan ekstrak di folder ini!")
    sys.exit(1)

# Load adapter
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Adapter loaded!")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================
# Step 3: Merge Model
# ============================================================
print("\n" + "="*50)
print("Step 3: Merging adapter dengan base model...")
print("="*50)

# Merge adapter ke base model
merged_model = model.merge_and_unload()

print("Merge selesai!")

# ============================================================
# Step 4: Simpan Merged Model
# ============================================================
print("\n" + "="*50)
print("Step 4: Menyimpan merged model...")
print("="*50)

os.makedirs(OUTPUT_DIR, exist_ok=True)

merged_model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

# Cek ukuran
total_size = 0
for f in os.listdir(OUTPUT_DIR):
    fpath = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        total_size += size
        print(f"  {f}: {size / 1024 / 1024:.2f} MB")

print(f"\nTotal size: {total_size / 1024 / 1024:.2f} MB")
print(f"Merged model disimpan di: {OUTPUT_DIR}")

# ============================================================
# Step 5: Convert ke GGUF (untuk llama.cpp)
# ============================================================
print("\n" + "="*50)
print("Step 5: Convert ke GGUF format...")
print("="*50)

print("""
Untuk convert ke GGUF, jalankan perintah ini di terminal:

# Install llama.cpp (jika belum)
pip install llama-cpp-python[server]

# Convert ke GGUF (pakai script dari llama.cpp)
python -m llama_cpp.llama_cpp convert \\
    --outfile {output} \\
    --outtype q4_k_m \\
    {model_dir}

Atau gunakan tool online:
https://huggingface.co/spaces/gguf-community/gguf-my-lama

Setelah dapat file .gguf, copy ke folder: E:\\App\\SWA\\AI\\models\\
""".format(output=GGUF_OUTPUT, model_dir=OUTPUT_DIR))

# ============================================================
# Step 6: Test Model (Optional)
# ============================================================
print("\n" + "="*50)
print("Step 6: Test model (opsional)...")
print("="*50)

test = input("Mau test model sekarang? (y/n): ").lower()
if test == 'y':
    print("\nTesting model...")
    
    # Test prompt
    messages = [
        {"role": "system", "content": "Kamu adalah asisten customer service yang profesional dan ramah."},
        {"role": "user", "content": "Saya mau tanya, pesanan saya belum sampai."}
    ]
    
    # Format sesuai template Qwen
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = merged_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print("\n" + "-"*50)
    print("PROMPT:")
    print(f"System: {messages[0]['content']}")
    print(f"User: {messages[1]['content']}")
    print("\nRESPONSE:")
    print(response)
    print("-"*50)

print("\n" + "="*50)
print("SELESAI!")
print("="*50)
print("""
Langkah terakhir:
1. Convert merged model ke GGUF (lihat Step 5)
2. Copy file .gguf ke: E:\\App\\SWA\\AI\\models\\
3. Update start_server.bat untuk pakai model baru
4. Jalankan server dan test!
""")
