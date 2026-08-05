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

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import os
import logging
import re
from datetime import datetime
from rag_engine import get_rag_engine
from db_engine import execute_query, get_schema_context
from sql_templates import get_few_shot_prompt

LOG_DIR = r"E:\App\SWA\AI\logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "chat_api.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("chat_api")

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
    system_prompt: Optional[str] = "Kamu adalah Business Intelligence AI yang membantu analisis data perusahaan. Tampilkan data dengan format yang rapi. Selalu jawab dengan data yang ada, jangan menolak."
    user_name: Optional[str] = None
    max_tokens: Optional[int] = 150
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
    logger.info(f"CREATE room={room_id}")
    if room_id in rooms or os.path.exists(get_room_path(room_id)):
        return {"status": "exists", "room_id": room_id}
    rooms[room_id] = [{"role": "system", "content": system_prompt}]
    save_room(room_id)
    return {"status": "created", "room_id": room_id}


@app.delete("/v1/chat/rooms/{room_id}")
async def delete_room(room_id: str):
    logger.info(f"DELETE room={room_id}")
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
async def send_message(room_id: str, req: ChatRequest, request: Request):
    client_ip = request.client.host
    logger.info(f"[{client_ip}] POST /v1/chat/{room_id}/messages | msg={req.message[:80]}")

    history = load_room(room_id)

    if not history:
        sys_content = req.system_prompt
        if req.user_name:
            sys_content += f" Pelanggan bernama {req.user_name}."
        history.append({"role": "system", "content": sys_content})

    # Auto-extract user_name dari pesan user ("nama saya X")
    skip_words = ["siapa", "apa", "dimana", "kapan", "mengapa", "bagaimana", "berapa", "yang", "ini", "itu"]
    name_match = re.search(r"nama saya (\w+)", req.message, re.IGNORECASE)
    if name_match:
        extracted_name = name_match.group(1).title()
        if extracted_name.lower() not in skip_words and len(extracted_name) > 2:
            sys_msg = history[0]
            if extracted_name not in sys_msg.get("content", ""):
                sys_msg["content"] += f" Pelanggan bernama {extracted_name}."
                logger.info(f"[{client_ip}] Extracted name: {extracted_name}")

    # Update system prompt jika user_name baru dikirim
    if req.user_name and history:
        sys_msg = history[0]
        if req.user_name not in sys_msg.get("content", ""):
            sys_msg["content"] += f" Pelanggan bernama {req.user_name}."

    history.append({"role": "user", "content": req.message})

    # RAG: search konteks relevan
    rag = get_rag_engine()
    rag.add_message(room_id, "user", req.message, len(history) - 1)

    # Detect if user is asking about data/database
    db_keywords = ["data", "berapa", "jumlah", "total", "list", "tampilkan", "lihat",
                   "member", "user", "pegawai", "transaksi", "penjualan", "omset", "order",
                   "tabel", "table", "database", "db", "sql", "select", "departemen",
                   "outlet", "shift", "menu", "log", "aktivitas", "daftar", "siapa saja"]
    is_db_query = any(kw in req.message.lower() for kw in db_keywords)

    db_result = None
    sql_failed = False  # Track if SQL generation was attempted but failed
    if is_db_query:
        # Build few-shot prompt with exact table/column names + examples
        # Returns None if query is unsupported (table doesn't exist in DB)
        few_shot_prompt = get_few_shot_prompt(req.message)
        if few_shot_prompt is None:
            logger.info(f"[{client_ip}] Query not supported by DB schema, skipping SQL gen")
            sql_failed = True  # Tell LLM that data is not available
        else:
            sql_prompt = [{"role": "user", "content": few_shot_prompt}]
            try:
                t_sql = datetime.now()
                async with httpx.AsyncClient(timeout=60.0) as client:
                    sql_resp = await client.post(
                        f"{LLAMA_SERVER}/v1/chat/completions",
                        json={
                            "model": "qwen2.5-cs-assistant",
                            "messages": sql_prompt,
                            "max_tokens": 80,
                            "temperature": 0.1,
                            "stop": ["\n\n", "Q:", "--"],
                        }
                    )
                if sql_resp.status_code == 200:
                    raw_sql = sql_resp.json()["choices"][0]["message"]["content"].strip()
                    logger.info(f"[{client_ip}] Raw SQL response: {raw_sql[:150]}")
                    sql_match = re.search(r"```(?:sql)?\s*(SELECT.*?)```", raw_sql, re.IGNORECASE | re.DOTALL)
                    if sql_match:
                        generated_sql = sql_match.group(1).strip()
                    elif raw_sql.upper().startswith("SELECT"):
                        generated_sql = raw_sql.split(";")[0].strip()
                    else:
                        sel_match = re.search(r"(SELECT\s+.+?)(?:;|$)", raw_sql, re.IGNORECASE | re.DOTALL)
                        generated_sql = sel_match.group(1).strip() if sel_match else None

                    if generated_sql:
                        logger.info(f"[{client_ip}] Generated SQL ({(datetime.now()-t_sql).total_seconds():.1f}s): {generated_sql}")
                        db_result = execute_query(generated_sql)
                        if db_result and db_result.get("success"):
                            logger.info(f"[{client_ip}] DB query OK: {db_result.get('row_count', 0)} rows")
                        else:
                            logger.warning(f"[{client_ip}] DB query failed: {db_result.get('error') if db_result else 'unknown'}")
                            sql_failed = True
                    else:
                        logger.warning(f"[{client_ip}] LLM did not return valid SQL: {raw_sql[:100]}")
                        sql_failed = True
                else:
                    logger.error(f"[{client_ip}] SQL gen HTTP error: {sql_resp.status_code}")
                    sql_failed = True
            except Exception as e:
                logger.error(f"[{client_ip}] SQL generation error: {type(e).__name__}: {e}")
                sql_failed = True
    
    # Search chat history RAG
    relevant_context = []
    keywords = ["nama", "siapa", "pesan", "order", "refund", "barang"]
    if any(kw in req.message.lower() for kw in keywords):
        relevant_context = rag.search_all_rooms(req.message, top_k=2)
    
    context_text = ""
    if relevant_context:
        context_lines = []
        for ctx in relevant_context:
            if ctx["room_id"] == room_id:
                context_lines.append(f"{ctx['role']}: {ctx['content']}")
            else:
                context_lines.append(f"[Room {ctx['room_id']}] {ctx['role']}: {ctx['content']}")
        context_text = "\n".join(context_lines)

    # Kirim 3 pesan terakhir saja
    MAX_HISTORY = 3
    sys_msgs = [m for m in history if m["role"] == "system"][:1]
    other_msgs = [m for m in history if m["role"] != "system"]
    recent = sys_msgs + other_msgs[-MAX_HISTORY:]

    # If it was a DB query but failed/unsupported, return directly without LLM
    # (prevents hallucination on data questions)
    if is_db_query and sql_failed:
        reply = "Maaf, data tersebut tidak tersedia di database kami saat ini."
        history.append({"role": "assistant", "content": reply})
        save_room(room_id)
        rag.add_message(room_id, "assistant", reply, len(history) - 1)
        logger.info(f"[{client_ip}] DB query failed/unsupported, returning safe reply")
        return {
            "room_id": room_id,
            "reply": reply,
            "history_count": len([m for m in history if m["role"] != "system"]),
            "rag_context_count": 0,
            "db_query": None,
            "db_rows": None,
        }

    # Build messages dengan context
    messages_for_llm = []

    # If DB query succeeded, format answer directly (skip second LLM call for speed)
    if db_result and db_result.get("success"):
        rows = db_result["rows"]
        columns = db_result["columns"]
        row_count = db_result["row_count"]

        # Format result as natural language
        if row_count == 0:
            reply = "Tidak ada data yang ditemukan."
        elif row_count == 1 and len(columns) == 1:
            # Single value result (e.g. COUNT)
            val = list(rows[0].values())[0]
            reply = f"{columns[0].replace('_', ' ').title()}: **{val}**"
        else:
            lines = [f"Ditemukan {row_count} data:\n"]
            for i, row in enumerate(rows[:10], 1):
                # Format each row as "key: value, key: value"
                parts = [f"{v}" for v in row.values()]
                lines.append(f"{i}. {' | '.join(parts)}")
            if row_count > 10:
                lines.append(f"... dan {row_count - 10} data lainnya.")
            reply = "\n".join(lines)

        history.append({"role": "assistant", "content": reply})
        save_room(room_id)
        rag.add_message(room_id, "assistant", reply, len(history) - 1)
        logger.info(f"[{client_ip}] DB reply (direct format) | rows={row_count}")
        return {
            "room_id": room_id,
            "reply": reply,
            "history_count": len([m for m in history if m["role"] != "system"]),
            "rag_context_count": len(relevant_context),
            "db_query": db_result.get("sql"),
            "db_rows": row_count,
        }

    # Add chat history context to system prompt
    sys_content = recent[0]["content"]
    if sql_failed:
        sys_content += "\n\nCATATAN: Gagal mengambil data dari database. Jangan menebak angka atau data. Sampaikan bahwa data tidak tersedia saat ini."
    if context_text:
        sys_content += f"\n\nIngat informasi penting dari memori:\n{context_text}"

    messages_for_llm.append({"role": "system", "content": sys_content})
    messages_for_llm.extend(recent[1:])

    payload = {
        "model": "qwen2.5-cs-assistant",
        "messages": messages_for_llm,
        "max_tokens": min(req.max_tokens, 150),
        "temperature": req.temperature,
    }

    try:
        t0 = datetime.now()
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{LLAMA_SERVER}/v1/chat/completions", json=payload)
            
            if resp.status_code != 200:
                error_detail = resp.text
                logger.error(f"[{client_ip}] LLM error {resp.status_code}: {error_detail[:200]}")
                raise HTTPException(status_code=resp.status_code, detail=f"LLM error: {error_detail}")
            
            data = resp.json()

        elapsed = (datetime.now() - t0).total_seconds()
        reply = data["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": reply})
        save_room(room_id)

        # Simpan reply ke RAG juga
        rag.add_message(room_id, "assistant", reply, len(history) - 1)

        logger.info(f"[{client_ip}] Reply OK ({elapsed:.1f}s) | tokens_used={data.get('usage', {}).get('total_tokens', '?')}")
        return {
            "room_id": room_id,
            "reply": reply,
            "history_count": len([m for m in history if m["role"] != "system"]),
            "rag_context_count": len(relevant_context),
            "db_query": db_result.get("sql") if db_result else None,
            "db_rows": db_result.get("row_count") if db_result else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{client_ip}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/chat/logs")
async def get_logs(lines: int = 50):
    log_file = os.path.join(LOG_DIR, "chat_api.log")
    if not os.path.exists(log_file):
        return {"logs": []}
    with open(log_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    return {"logs": [l.rstrip() for l in all_lines[-lines:]], "total": len(all_lines)}


@app.get("/v1/chat/rag/stats")
async def rag_stats():
    rag = get_rag_engine()
    return rag.get_stats()


@app.get("/v1/chat/rag/search")
async def rag_search(q: str, room_id: Optional[str] = None, top_k: int = 5):
    rag = get_rag_engine()
    if room_id:
        results = rag.search(q, room_id=room_id, top_k=top_k)
    else:
        results = rag.search_all_rooms(q, top_k=top_k)
    return {"query": q, "results": results}


@app.get("/v1/chat/db/query")
async def db_query(sql: str):
    """Execute SQL query langsung (READ ONLY)"""
    result = execute_query(sql)
    return result


@app.get("/v1/chat/db/schema")
async def db_schema(table: Optional[str] = None):
    """Get DB schema info"""
    rag = get_rag_engine()
    if table:
        results = rag.search_db_schema(table, top_k=1)
    else:
        results = rag.search_db_schema("all tables", top_k=5)
    return {"schema_context": results}


if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 50)
    logger.info("Room Chat API Starting (with RAG + DB Engine)")
    logger.info(f"llama.cpp server: {LLAMA_SERVER}")
    logger.info(f"Data folder: {DATA_DIR}")
    logger.info(f"Log file: {LOG_DIR}/chat_api.log")
    rag = get_rag_engine()
    stats = rag.get_stats()
    logger.info(f"RAG Engine: {stats['total_documents']} chat docs, {stats['db_schema_chunks']} DB schema chunks")
    logger.info("DB Engine: PostgreSQL (dev_richz) - READ ONLY")
    logger.info("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8001)
