# Cell baru di Colab - Convert ke GGUF
# Jalankan SETELAH merge selesai

!git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp 2>/dev/null || true

!python /tmp/llama.cpp/convert_hf_to_gguf.py \
    --outfile qwen2.5-cs-assistant.gguf \
    --outtype q4_k_m \
    ./qwen2.5-cs-assistant-merged

from google.colab import files
import os

gguf = "qwen2.5-cs-assistant.gguf"
if os.path.exists(gguf):
    print(f"GGUF: {os.path.getsize(gguf)/1024/1024:.1f} MB")
    files.download(gguf)
