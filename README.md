# AI Chat Server

Local AI chat server with fine-tuned Qwen2.5 model, room-based conversations, and history persistence.

## Features

- **Fine-tuned Qwen2.5-1.5B** model for Customer Service & Director Assistant
- **Room-based chat** with conversation history
- **OpenAI-compatible API** format
- **Multi-request support** with concurrent processing
- **Network accessible** - access from any device on same network

## Quick Start

```bash
# Clone
git clone https://github.com/jhustinn/AI_Chat.git
cd AI_Chat

# Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install llama-cpp-python[server] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install fastapi uvicorn httpx huggingface-hub

# Download model
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct-GGUF', filename='qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='./models')"

# Run
start_server.bat
start_chat_server.bat
```

## Documentation

- [Setup Guide](SETUP_GUIDE.md) - Complete setup instructions
- [API Docs](API_DOCS.md) - API endpoints reference
- [Colab Guide](finetune/GUIDE_COLAB.md) - Fine-tuning guide

## API Endpoints

### Room Chat API (Port 8001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/rooms/{room_id}` | Create room |
| `POST` | `/v1/chat/{room_id}/messages` | Send message |
| `GET` | `/v1/chat/{room_id}/history` | Get history |
| `GET` | `/v1/chat/rooms` | List rooms |
| `DELETE` | `/v1/chat/rooms/{room_id}` | Delete room |

### llama.cpp Server (Port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion |
| `GET` | `/v1/models` | List models |

## Tech Stack

- **LLM:** Qwen2.5-1.5B (GGUF format)
- **Inference:** llama.cpp / llama-cpp-python
- **API:** FastAPI + Uvicorn
- **Fine-tuning:** QLoRA (Google Colab)
- **Model:** https://drive.google.com/file/d/1-HaEJUVZZ5Q1MfxY_y7Rykw1MCqDHnnz/view?usp=sharing
