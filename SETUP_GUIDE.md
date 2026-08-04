# AI Chat Server - Setup Guide

> Complete guide to set up and run the AI Chat Server with fine-tuned Qwen2.5 model on a new machine.

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Detailed Setup](#detailed-setup)
6. [Fine-Tuning Guide](#fine-tuning-guide)
7. [API Documentation](#api-documentation)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This project runs a local AI chat server with:

- **Qwen2.5-1.5B** model (fine-tuned for Customer Service & Director Assistant)
- **llama.cpp** server for LLM inference
- **Room Chat API** for multi-room conversations with history persistence
- **OpenAI-compatible API** format

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Client (Postman, Browser, App)                         │
│  http://localhost:8001/v1/chat/{room_id}/messages       │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  Room Chat API (FastAPI) - Port 8001                    │
│  - Conversation history per room                        │
│  - Multi-room support                                   │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  llama.cpp Server - Port 8000                           │
│  - Qwen2.5-1.5B model (fine-tuned)                      │
│  - OpenAI-compatible API                                │
└─────────────────────────────────────────────────────────┘
```

---

## System Requirements

### Minimum

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11 (64-bit) |
| RAM | 8 GB |
| Storage | 10 GB free |
| CPU | Any modern x86_64 |
| Python | 3.10+ |

### Recommended (for Fine-Tuning)

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA GTX 1650+ (4GB+ VRAM) |
| RAM | 16 GB |
| Storage | 20 GB free |
| Internet | Stable connection for model download |

---

## Project Structure

```
AI_Chat/
├── README.md                          # This file
├── SETUP_GUIDE.md                     # Setup documentation
├── chat_server.py                     # Room Chat API server
├── start_server.bat                   # Start llama.cpp server
├── start_chat_server.bat              # Start Room Chat API
├── test_multi.py                      # Multi-request test script
├── API_DOCS.md                        # API documentation
│
├── models/                            # Model files (not in git)
│   ├── qwen2.5-1.5b-instruct-q4_k_m.gguf   # Base model
│   └── qwen2.5-cs-assistant.gguf            # Fine-tuned model
│
├── finetune/                          # Fine-tuning scripts
│   ├── dataset_example.json           # Training dataset (15 examples)
│   ├── train_colab.py                 # Training script for Google Colab
│   ├── merge_colab_cpu.py             # Merge adapter in Colab (CPU)
│   ├── convert_simple.py              # Convert to GGUF format
│   ├── GUIDE_COLAB.md                 # Colab training guide
│   └── qwen2.5-cs-assistant-adapter/  # LoRA adapter (not in git)
│
├── data/                              # Runtime data (not in git)
│   └── rooms/                         # Conversation history
│
└── venv/                              # Python virtual environment (not in git)
```

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/jhustinn/AI_Chat.git
cd AI_Chat
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

### 4. Install Dependencies

```bash
pip install llama-cpp-python[server] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install fastapi uvicorn httpx huggingface-hub
```

### 5. Download Model

```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Qwen/Qwen2.5-1.5B-Instruct-GGUF',
    filename='qwen2.5-1.5b-instruct-q4_k_m.gguf',
    local_dir='./models'
)
"
```

### 6. Start Servers

```bash
# Terminal 1 - llama.cpp server
start_server.bat

# Terminal 2 - Room Chat API
start_chat_server.bat
```

### 7. Access

- **Room Chat API Docs:** http://localhost:8001/docs
- **llama.cpp Docs:** http://localhost:8000/docs
- **Network Access:** http://<your-ip>:8001/docs

---

## Detailed Setup

### Step 1: Install Python

Download Python 3.12+ from https://www.python.org/downloads/

**Important:** Check "Add Python to PATH" during installation.

Verify:
```bash
python --version
pip --version
```

### Step 2: Clone Repository

```bash
git clone https://github.com/jhustinn/AI_Chat.git
cd AI_Chat
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
# llama.cpp server with CPU support
pip install llama-cpp-python[server] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Web framework
pip install fastapi uvicorn httpx

# Hugging Face (for model download)
pip install huggingface-hub
```

### Step 6: Create Models Directory

```bash
mkdir models
```

### Step 7: Download Models

Models tersedia di Google Drive:

1. Download `models.zip` dari Google Drive
2. Ekstrak ke folder `models/` di project

**ATAU download manual dari Hugging Face:**
```bash
mkdir models
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Qwen/Qwen2.5-1.5B-Instruct-GGUF',
    filename='qwen2.5-1.5b-instruct-q4_k_m.gguf',
    local_dir='./models'
)
"
```

### Step 9: Configure Server Scripts

**start_server.bat:**
```batch
@echo off
echo Starting llama.cpp server...
venv\Scripts\python.exe -m llama_cpp.server ^
  --model models\qwen2.5-1.5b-instruct-q4_k_m.gguf ^
  --n_ctx 512 ^
  --n_gpu_layers 0 ^
  --chat_format chatml ^
  --host 0.0.0.0 ^
  --port 8000
```

**start_chat_server.bat:**
```batch
@echo off
echo Starting Room Chat API...
venv\Scripts\python.exe chat_server.py
```

### Step 10: Start Servers

```bash
# Terminal 1
start_server.bat

# Wait for model to load (~60 seconds)

# Terminal 2
start_chat_server.bat
```

### Step 11: Verify

```bash
# Check llama.cpp server
curl http://localhost:8000/docs

# Check Room Chat API
curl http://localhost:8001/
```

---

## Fine-Tuning Guide

### Overview

Fine-tuning uses **QLoRA** (Quantized Low-Rank Adaptation) to customize the model for your specific use case.

### Step 1: Prepare Dataset

Edit `finetune/dataset_example.json` with your training data:

```json
[
  {
    "messages": [
      {"role": "system", "content": "Kamu adalah customer service yang profesional."},
      {"role": "user", "content": "Contoh pertanyaan user"},
      {"role": "assistant", "content": "Contoh jawaban yang diinginkan"}
    ]
  }
]
```

**Tips:**
- Minimum 50 examples for basic results
- 200-500 examples for good results
- 1000+ examples for optimal results
- Include variety in questions and responses

### Step 2: Upload to Google Colab

1. Open https://colab.research.google.com
2. Create new notebook
3. Enable GPU: Runtime > Change runtime type > T4 GPU
4. Upload `finetune/train_colab.py` content
5. Upload `finetune/dataset_example.json`

### Step 3: Run Training

Execute cells in order:
1. Cell 1: Install dependencies (auto-restarts runtime)
2. Cell 2-6: Import, load model, configure LoRA
3. Cell 7-8: Training arguments and trainer
4. Cell 9: Start training (~15-30 minutes)
5. Cell 10: Save adapter

### Step 4: Download Adapter

After training completes:
```python
from google.colab import files
import shutil

shutil.make_archive('qwen2.5-cs-assistant-adapter', 'zip', '.', 'qwen2.5-cs-assistant-adapter')
files.download('qwen2.5-cs-assistant-adapter.zip')
```

### Step 5: Merge Adapter (in Colab)

```python
# Paste in new cell
from google.colab import drive
import shutil, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

!pip install -q torchao peft --upgrade

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

model = PeftModel.from_pretrained(model, "./qwen2.5-cs-assistant-adapter")
merged = model.merge_and_unload()

merged.save_pretrained("./qwen2.5-cs-assistant-merged", safe_serialization=True)
tokenizer.save_pretrained("./qwen2.5-cs-assistant-merged")
```

### Step 6: Convert to GGUF

```python
!git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp 2>/dev/null || true
!pip install -q gguf sentencepiece protobuf

!python /tmp/llama.cpp/convert_hf_to_gguf.py \
    --outfile qwen2.5-cs-assistant.gguf \
    --outtype q8_0 \
    ./qwen2.5-cs-assistant-merged
```

### Step 7: Download GGUF

```python
from google.colab import files, drive
import shutil

# Option A: Direct download (smaller files)
files.download('qwen2.5-cs-assistant.gguf')

# Option B: Via Google Drive (if direct fails)
drive.mount('/content/drive')
shutil.copy('qwen2.5-cs-assistant.gguf', '/content/drive/MyDrive/qwen2.5-cs-assistant.gguf')
# Download from https://drive.google.com/drive/my-drive
```

### Step 8: Deploy Fine-Tuned Model

1. Copy `qwen2.5-cs-assistant.gguf` to `models/` folder
2. Update `start_server.bat`:
   ```batch
   --model models\qwen2.5-cs-assistant.gguf
   ```
3. Restart server

---

## API Documentation

### Room Chat API (Port 8001)

#### Create Room
```http
POST /v1/chat/rooms/{room_id}
Content-Type: application/json

{
  "system_prompt": "Kamu adalah customer service yang profesional."
}
```

#### Send Message
```http
POST /v1/chat/{room_id}/messages
Content-Type: application/json

{
  "message": "Saya butuh bantuan",
  "max_tokens": 200,
  "temperature": 0.7
}
```

#### Get History
```http
GET /v1/chat/{room_id}/history
```

#### List Rooms
```http
GET /v1/chat/rooms
```

#### Delete Room
```http
DELETE /v1/chat/rooms/{room_id}
```

### llama.cpp Server (Port 8000)

#### Chat Completion
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "qwen2.5-cs-assistant",
  "messages": [
    {"role": "system", "content": "Kamu adalah asisten AI."},
    {"role": "user", "content": "Halo!"}
  ],
  "max_tokens": 200,
  "temperature": 0.7
}
```

#### List Models
```http
GET /v1/models
```

### Response Format

```json
{
  "room_id": "test-room",
  "reply": "AI response here...",
  "history_count": 4
}
```

---

## Network Access

To access from other devices on the same network:

1. Find your local IP:
   ```bash
   ipconfig
   ```

2. Access from other device:
   ```
   http://<your-ip>:8001/docs
   ```

3. Ensure Windows Firewall allows ports 8000 and 8001.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'llama_cpp'"
```bash
pip install llama-cpp-python[server] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### "CUDA not available" (Expected for CPU mode)
This is normal if you don't have an NVIDIA GPU or CUDA is not installed. The server will run on CPU.

### "Out of Memory" when loading model
Reduce context window in `start_server.bat`:
```batch
--n_ctx 256
```

### Server takes long to start
Normal on first run. Model loading takes 30-120 seconds depending on hardware.

### "Port already in use"
```bash
# Find process using port
netstat -ano | findstr :8000

# Kill process
taskkill /PID <process_id> /F
```

### Download fails for model
```bash
# Set Hugging Face cache to different drive
set HF_HOME=E:\huggingface_cache
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(...)"
```

### C: drive full during pip install
```bash
# Install to different target
pip install <package> --target E:\App\SWA\AI\venv\Lib\site-packages --no-cache-dir
```

---

## Updating

### Pull latest changes
```bash
git pull origin main
```

### Update dependencies
```bash
pip install --upgrade llama-cpp-python[server] fastapi uvicorn httpx
```

---

## License

This project is for educational and personal use.

---

## Links

- [Qwen2.5 Model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Google Colab](https://colab.research.google.com)
