# Convert GGUF - Complete
# Jalankan di cell baru Colab

import os

# Download script convert + install gguf
!wget -q https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py
!pip install -q gguf

# Convert ke Q8_0
!python convert_hf_to_gguf.py \
    --outfile qwen2.5-cs-assistant.gguf \
    --outtype q8_0 \
    ./qwen2.5-cs-assistant-merged

# Download
from google.colab import files
gguf = "qwen2.5-cs-assistant.gguf"
if os.path.exists(gguf):
    print(f"GGUF: {os.path.getsize(gguf)/1024/1024:.1f} MB")
    files.download(gguf)
else:
    print("Gagal! Cek error di atas.")
