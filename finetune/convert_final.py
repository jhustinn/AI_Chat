# Convert GGUF - FINAL FIX
import os

# Clone repo lengkap (butuh file conversion.py dll)
!git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp 2>/dev/null || true

# Install dependencies
!pip install -q gguf sentencepiece protobuf

# Convert
!python /tmp/llama.cpp/convert_hf_to_gguf.py \
    --outfile /content/qwen2.5-cs-assistant.gguf \
    --outtype q8_0 \
    /content/qwen2.5-cs-assistant-merged

# Download
from google.colab import files
gguf = "/content/qwen2.5-cs-assistant.gguf"
if os.path.exists(gguf):
    print(f"GGUF: {os.path.getsize(gguf)/1024/1024:.1f} MB")
    files.download(gguf)
