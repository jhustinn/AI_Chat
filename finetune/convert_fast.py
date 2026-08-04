# Convert GGUF - TANPA install llama-cpp-python (lebih cepat!)
# Jalankan di cell baru Colab

import os

# Download script convert langsung (tidak perlu clone repo)
!wget -q https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py
!wget -q https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf_qwen.py 2>/dev/null || true

# Install dependencies ringan saja
!pip install -q sentencepiece protobuf

# Convert
!python convert_hf_to_gguf.py \
    --outfile qwen2.5-cs-assistant.gguf \
    --outtype q4_k_m \
    ./qwen2.5-cs-assistant-merged

# Download
from google.colab import files
gguf = "qwen2.5-cs-assistant.gguf"
if os.path.exists(gguf):
    print(f"\nGGUF size: {os.path.getsize(gguf)/1024/1024:.1f} MB")
    files.download(gguf)
else:
    print("ERROR: Convert gagal!")
