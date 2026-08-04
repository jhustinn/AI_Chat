# llama.cpp Server API Documentation

**Base URL:** `http://localhost:8000`
**Format:** OpenAI-compatible API
**Docs:** http://localhost:8000/docs

---

## 1. Chat Completions (Paling Umum)

### POST `/v1/chat/completions`

**Request Body:**
```json
{
  "model": "qwen2.5-1.5b",
  "messages": [
    {"role": "system", "content": "Kamu adalah asisten AI yang helpful."},
    {"role": "user", "content": "Apa itu Python?"}
  ],
  "max_tokens": 200,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-xxxxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "qwen2.5-1.5b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Python adalah bahasa pemrograman..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 50,
    "total_tokens": 75
  }
}
```

**Response (streaming, stream=true):**
```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Python"},"index":0}]}

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":" adalah"},"index":0}]}

data: [DONE]
```

---

## 2. Text Completion

### POST `/v1/completions`

**Request Body:**
```json
{
  "model": "qwen2.5-1.5b",
  "prompt": "def fibonacci(n):\n",
  "max_tokens": 150,
  "temperature": 0.3,
  "stop": ["\n\n"]
}
```

**Response:**
```json
{
  "id": "cmpl-xxxxx",
  "object": "text_completion",
  "choices": [
    {
      "text": "    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 30,
    "total_tokens": 40
  }
}
```

---

## 3. Embeddings

### POST `/v1/embeddings`

**Request Body:**
```json
{
  "model": "qwen2.5-1.5b",
  "input": "Hello world"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0023, -0.0091, 0.0156, ...]
    }
  ],
  "model": "qwen2.5-1.5b",
  "usage": {
    "prompt_tokens": 3,
    "total_tokens": 3
  }
}
```

---

## 4. List Models

### GET `/v1/models`

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen2.5-1.5b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local"
    }
  ]
}
```

---

## Parameter Reference

### Messages Role
| Role | Deskripsi |
|------|-----------|
| `system` | Instruksi untuk AI (opsional) |
| `user` | Input dari user |
| `assistant` | Response dari AI (untuk multi-turn) |

### Parameters
| Parameter | Type | Default | Deskripsi |
|-----------|------|---------|-----------|
| `model` | string | - | Nama model (bebas, server pakai 1 model) |
| `messages` | array | - | Array of message objects |
| `max_tokens` | integer | -1 | Max token yang di-generate (-1 = sampai habis) |
| `temperature` | float | 0.8 | Kreativitas (0.0 = deterministik, 2.0 = sangat random) |
| `top_p` | float | 0.9 | Nucleus sampling |
| `top_k` | integer | 40 | Top-K sampling |
| `stream` | boolean | false | true = streaming response |
| `stop` | array | null | Stop sequences (misal: ["\n", "User:"]) |
| `frequency_penalty` | float | 0.0 | Penalti untuk token yang sering muncul |
| `presence_penalty` | float | 0.0 | Penalti untuk token yang sudah ada |
| `seed` | integer | null | Seed untuk reproducibility |

---

## Contoh di Postman

### Chat Completion
```
Method: POST
URL: http://localhost:8000/v1/chat/completions
Headers: Content-Type: application/json
Body (raw JSON):

{
  "model": "qwen2.5-1.5b",
  "messages": [
    {"role": "system", "content": "Jawab dalam Bahasa Indonesia, singkat dan jelas."},
    {"role": "user", "content": "Apa perbedaan list dan tuple di Python?"}
  ],
  "max_tokens": 200,
  "temperature": 0.7
}
```

### Multi-turn Conversation
```
{
  "model": "qwen2.5-1.5b",
  "messages": [
    {"role": "system", "content": "Kamu adalah guru programming."},
    {"role": "user", "content": "Apa itu variable?"},
    {"role": "assistant", "content": "Variable adalah tempat menyimpan data..."},
    {"role": "user", "content": "Beri contoh di Python"}
  ],
  "max_tokens": 150
}
```

### Streaming (di Postman pakai tab "Send" lalu lihat response real-time)
```
{
  "model": "qwen2.5-1.5b",
  "messages": [
    {"role": "user", "content": "Ceritakan dongeng singkat"}
  ],
  "max_tokens": 300,
  "stream": true
}
```

---

## Error Codes

| Code | Deskripsi |
|------|-----------|
| 200 | Success |
| 422 | Validation Error (parameter salah) |
| 500 | Server Error |

---

## Tips untuk GTX 1650

- **max_tokens**: Jangan terlalu besar (200-500 cukup)
- **n_ctx**: Server di-set 2048 token (context window)
- **Concurrent**: Bisa handle ~3-5 request bersamaan (CPU mode)
- **Response time**: 3-20 detik per request tergantung panjang output
