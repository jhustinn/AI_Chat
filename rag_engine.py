"""
RAG Engine - Vector store untuk room chat memory + DB knowledge base
Menggunakan ChromaDB + ONNXMiniLM_L6_V2 embedding
"""

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
import os
import json
from datetime import datetime

CHROMA_DIR = r"E:\App\SWA\AI\data\chroma_db"
ROOMS_DIR = r"E:\App\SWA\AI\data\rooms"
DB_KNOWLEDGE_DIR = r"E:\App\SWA\AI\data\db_knowledge"

os.makedirs(CHROMA_DIR, exist_ok=True)

# Global embedding function (load sekali saja)
_ef = ONNXMiniLM_L6_V2()


class RAGEngine:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        # Collection untuk chat history
        self.collection = self.client.get_or_create_collection(
            name="chat_history",
            metadata={"hnsw:space": "cosine"},
            embedding_function=_ef,
        )
        
        # Collection untuk DB schema
        self.db_collection = self.client.get_or_create_collection(
            name="db_schema",
            metadata={"hnsw:space": "cosine"},
            embedding_function=_ef,
        )
        
        self._index_existing_rooms()
        self._index_db_schema()

    def _index_existing_rooms(self):
        """Index semua room JSON yang sudah ada"""
        if not os.path.exists(ROOMS_DIR):
            return

        for filename in os.listdir(ROOMS_DIR):
            if not filename.endswith(".json"):
                continue
            room_id = filename.replace(".json", "")
            self._index_room(room_id)

    def _index_db_schema(self):
        """Index DB schema untuk text-to-SQL"""
        knowledge_file = os.path.join(DB_KNOWLEDGE_DIR, "knowledge.txt")
        if not os.path.exists(knowledge_file):
            return
        
        # Check if already indexed
        existing = self.db_collection.get(ids=["db_schema_full"])
        if existing and existing["ids"]:
            return
        
        with open(knowledge_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split into chunks (max 1000 chars each)
        chunks = []
        current_chunk = ""
        for line in content.split("\n"):
            if len(current_chunk) + len(line) > 1000:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += "\n" + line if current_chunk else line
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Index chunks
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                self.db_collection.add(
                    ids=[f"db_schema_{i}"],
                    documents=[chunk],
                    metadatas=[{"source": "db_schema", "chunk_index": i}]
                )

    def _index_room(self, room_id: str):
        """Index semua pesan dalam satu room"""
        path = os.path.join(ROOMS_DIR, f"{room_id}.json")
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            messages = json.load(f)

        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                continue

            doc_id = f"{room_id}_{i}"
            content = msg["content"]

            # Skip jika sudah ada
            existing = self.collection.get(ids=[doc_id])
            if existing and existing["ids"]:
                continue

            self.collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[{
                    "room_id": room_id,
                    "role": msg["role"],
                    "index": i,
                    "timestamp": datetime.now().isoformat(),
                }]
            )

    def add_message(self, room_id: str, role: str, content: str, index: int):
        """Tambah pesan baru ke vector store"""
        doc_id = f"{room_id}_{index}"
        self.collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[{
                "room_id": room_id,
                "role": role,
                "index": index,
                "timestamp": datetime.now().isoformat(),
            }]
        )

    def search(self, query: str, room_id: str = None, top_k: int = 5) -> list:
        """Search pesan relevan dari semua room"""
        where_filter = None
        if room_id:
            where_filter = {"room_id": room_id}

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
        )

        context = []
        if results and results["documents"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                context.append({
                    "content": doc,
                    "room_id": meta["room_id"],
                    "role": meta["role"],
                })

        return context

    def search_all_rooms(self, query: str, top_k: int = 5) -> list:
        """Search dari SEMUA room (cross-room memory)"""
        return self.search(query, room_id=None, top_k=top_k)

    def search_db_schema(self, query: str, top_k: int = 3) -> list:
        """Search DB schema untuk menemukan table yang relevan"""
        results = self.db_collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        
        context = []
        if results and results["documents"]:
            for doc in results["documents"][0]:
                context.append(doc)
        
        return context

    def get_stats(self) -> dict:
        """Stats vector store"""
        chat_count = self.collection.count()
        db_count = self.db_collection.count()
        return {
            "total_documents": chat_count,
            "db_schema_chunks": db_count,
            "collection_name": "chat_history",
        }


# Singleton
_engine = None


def get_rag_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
