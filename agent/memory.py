"""Memory layer — ChromaDB (Ollama embeddings) + SQLite."""
from __future__ import annotations

import sqlite3
import time

import chromadb
import ollama as _ollama_client
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from . import config


class OllamaEmbed(EmbeddingFunction):
    """Ollama embedding function cho ChromaDB."""

    def __init__(self, model: str = "nomic-embed-text", host: str | None = None):
        self._client = _ollama_client.Client(host=(host or config.OLLAMA_HOST))
        self._model = model

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for text in input:
            resp = self._client.embeddings(model=self._model, prompt=text)
            out.append(resp["embedding"])
        return out


class VectorMemory:
    """ChromaDB — vector search voi Ollama embeddings."""

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._ef = OllamaEmbed()
        self._collection = self._client.get_or_create_collection(
            name="tech_knowledge",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, doc_id: str, text: str, metadata: dict):
        existing = self._collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            return
        self._collection.add(ids=[doc_id], documents=[text], metadatas=[metadata])

    def add_batch(self, ids: list[str], texts: list[str], metadatas: list[dict]):
        existing = self._collection.get(ids=ids)
        existing_ids = set(existing["ids"]) if existing["ids"] else set()
        nids, ntexts, nmetas = [], [], []
        for i, t, m in zip(ids, texts, metadatas):
            if i not in existing_ids:
                nids.append(i)
                ntexts.append(t)
                nmetas.append(m)
        if nids:
            self._collection.add(ids=nids, documents=ntexts, metadatas=nmetas)

    def search(self, query: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
        kwargs = {"query_texts": [query], "n_results": n_results}
        if where:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results["distances"] else 0,
            })
        return output

    def count(self) -> int:
        return self._collection.count()

    def get_categories(self) -> dict[str, int]:
        if self._collection.count() == 0:
            return {}
        all_data = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in all_data["metadatas"]:
            cat = meta.get("category", "unknown")
            counts[cat] = counts.get(cat, 0) + 1
        return counts


class StructuredMemory:
    """SQLite — code snippets, notes, chat history."""

    def __init__(self):
        self._db_path = config.DATA_DIR / "memory.db"
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS snippets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                code TEXT NOT NULL,
                language TEXT,
                category TEXT,
                tags TEXT,
                created_at REAL,
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                tags TEXT,
                created_at REAL,
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS bookmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                title TEXT,
                description TEXT,
                category TEXT,
                created_at REAL
            );
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                timestamp REAL
            );
        """)

    def add_snippet(self, title: str, code: str, language: str = "", category: str = "", tags: str = "") -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO snippets (title, code, language, category, tags, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (title, code, language, category, tags, now, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_snippets(self, category: str | None = None, limit: int = 20) -> list[dict]:
        if category:
            rows = self._conn.execute(
                "SELECT * FROM snippets WHERE category=? ORDER BY updated_at DESC LIMIT ?",
                (category, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM snippets ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def add_note(self, title: str, content: str, category: str = "", tags: str = "") -> int:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO notes (title, content, category, tags, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (title, content, category, tags, now, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def search_notes(self, query: str, limit: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM notes WHERE title LIKE ? OR content LIKE ? ORDER BY updated_at DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_chat(self, role: str, content: str, category: str = ""):
        self._conn.execute(
            "INSERT INTO chat_history (role, content, category, timestamp) VALUES (?,?,?,?)",
            (role, content, category, time.time()),
        )
        self._conn.commit()

    def get_chat_history(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM chat_history ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def stats(self) -> dict:
        snippets = self._conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
        notes = self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        chats = self._conn.execute("SELECT COUNT(*) FROM chat_history").fetchone()[0]
        return {"snippets": snippets, "notes": notes, "chat_history": chats}
