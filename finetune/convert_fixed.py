# Convert GGUF - Fixed version
# Jalankan di cell baru Colab

import os

# Convert ke GGUF format f16 dulu
!python convert_hf_to_gguf.py \
    --outfile qwen2.5-cs-assistant-f16.gguf \
    --outtype f16 \
    ./qwen2.5-cs-assistant-merged

# Quantize ke Q4_K_M (lebih kecil)
!git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp 2>/dev/null || true

# Build llama-quantize
!cd /tmp/llama.cpp && cmake -B build && cmake --build build --config Release -j4

# Quantize
!/tmp/llama.cpp/build/bin/llama-quantize \
    qwen2.5-cs-assistant-f16.gguf \
    qwen2.5-cs-assistant.gguf \
    Q4_K_M

# Download
from google.colab import files
gguf = "qwen2.5-cs-assistant.gguf"
f16 = "qwen2.5-cs-assistant-f16.gguf"

if os.path.exists(gguf):
    print(f"GGUF Q4_K_M: {os.path.getsize(gguf)/1024/1024:.1f} MB")
    files.download(gguf)
elif os.path.exists(f16):
    print(f"GGUF F16: {os.path.getsize(f16)/1024/1024:.1f} MB")
    print("Quantize gagal, download F16 saja (lebih besar tapi jalan)")
    files.download(f16)
