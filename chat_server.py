"""
Room-based Chat API dengan conversation history
Wrapper untuk llama.cpp server

Endpoints:
- POST /v1/chat/{room_id}/messages  - Kirim pesan ke room
- GET  /v1/chat/{room_id}/history   - Ambil history room
- GET  /v1/chat/rooms               - List semua room
- DELETE /v1/chat/{room_id}         - Hapus room
- POST /v1/chat/rooms               - Buat room baru
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import os
from datetime import datetime

app = FastAPI(title="Room Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
LLAMA_SERVER = "http://localhost:8000"
DATA_DIR = r"E:\App\SWA\AI\data\rooms"
os.makedirs(DATA_DIR, exist_ok=True)

# In-memory rooms cache
rooms: dict[str, list] = {}


class Message(BaseModel):
    role: str  # "user" atau "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = "Kamu adalah asisten AI yang helpful."
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7


class RoomInfo(BaseModel):
    room_id: str
    message_count: int
    last_message: Optional[str] = None
    created_at: str


def get_room_path(room_id: str) -> str:
    return os.path.join(DATA_DIR, f"{room_id}.json")


def load_room(room_id: str) -> list:
    if room_id in rooms:
        return rooms[room_id]
    path = get_room_path(room_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            rooms[room_id] = json.load(f)
            return rooms[room_id]
    rooms[room_id] = []
    return rooms[room_id]


def save_room(room_id: str):
    path = get_room_path(room_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rooms[room_id], f, ensure_ascii=False, indent=2)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Room Chat API running"}


@app.get("/v1/chat/rooms")
async def list_rooms():
    result = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".json"):
            rid = f.replace(".json", "")
            history = load_room(rid)
            last_msg = history[-1]["content"][:50] if history else None
            result.append({
                "room_id": rid,
                "message_count": len(history),
                "last_message": last_msg,
            })
    return {"rooms": result}


@app.post("/v1/chat/rooms/{room_id}")
async def create_room(room_id: str, system_prompt: Optional[str] = "Kamu adalah asisten AI yang helpful."):
    if room_id in rooms or os.path.exists(get_room_path(room_id)):
        return {"status": "exists", "room_id": room_id}
    rooms[room_id] = [{"role": "system", "content": system_prompt}]
    save_room(room_id)
    return {"status": "created", "room_id": room_id}


@app.delete("/v1/chat/rooms/{room_id}")
async def delete_room(room_id: str):
    path = get_room_path(room_id)
    if os.path.exists(path):
        os.remove(path)
    if room_id in rooms:
        del rooms[room_id]
    return {"status": "deleted", "room_id": room_id}


@app.get("/v1/chat/{room_id}/history")
async def get_history(room_id: str):
    history = load_room(room_id)
    return {
        "room_id": room_id,
        "messages": [m for m in history if m["role"] != "system"],
        "count": len([m for m in history if m["role"] != "system"]),
    }


@app.post("/v1/chat/{room_id}/messages")
async def send_message(room_id: str, req: ChatRequest):
    history = load_room(room_id)

    if not history:
        history.append({"role": "system", "content": req.system_prompt})

    history.append({"role": "user", "content": req.message})

    payload = {
        "model": "qwen2.5-cs-assistant",
        "messages": history,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{LLAMA_SERVER}/v1/chat/completions", json=payload)
            data = resp.json()

        reply = data["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": reply})
        save_room(room_id)

        return {
            "room_id": room_id,
            "reply": reply,
            "history_count": len([m for m in history if m["role"] != "system"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("="*50)
    print("Room Chat API")
    print(f"llama.cpp server: {LLAMA_SERVER}")
    print(f"Data folder: {DATA_DIR}")
    print("="*50)
    print("\nEndpoints:")
    print("  POST /v1/chat/{room_id}/messages  - Kirim pesan")
    print("  GET  /v1/chat/{room_id}/history   - Lihat history")
    print("  GET  /v1/chat/rooms               - List rooms")
    print("  POST /v1/chat/rooms/{room_id}     - Buat room")
    print("  DELETE /v1/chat/rooms/{room_id}   - Hapus room")
    print("\nDocs: http://localhost:8001/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)
