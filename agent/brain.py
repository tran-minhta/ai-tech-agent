"""Brain layer — RAG query pipeline + LLM response generation."""
from __future__ import annotations

import json
import re

import ollama

from . import config
from .memory import VectorMemory, StructuredMemory

SYSTEM_PROMPT = """Ban la AI Tech Knowledge Agent — mot chuyen gia ve lap trinh, OS, va cong nghe.
Ban co long-term memory luu tru knowledge va co the tra cuu offline.

Khi tra loi:
1. Su dung context tu knowledge base de tra loi chinh xac
2. Neu khong co context phu hop, tra loi tu kinh nghiem nhac la "day la kien thuc co ban"
3. Luon cho vi du code minh hoa khi hoi ve lap trinh
4. Tra loi bang tieng Viet neu nguoi dung hoi bang tieng Viet
5. Ngan gon, di thang vao van de

Neu nguoi dung muon luu note/snippet, tra ve JSON:
{"action": "save_note", "title": "...", "content": "...", "category": "..."}
{"action": "save_snippet", "title": "...", "code": "...", "language": "...", "category": "..."}

Neu nguoi dung hoi ve status, tra ve:
{"action": "status"}
"""


class Brain:
    """RAG pipeline — tim context tu memory, gui LLM, tra ve response."""

    def __init__(self, vector_mem: VectorMemory, struct_mem: StructuredMemory):
        self.vector = vector_mem
        self.structured = struct_mem
        self._client = None

    def _get_client(self) -> ollama.Client:
        if self._client is None:
            self._client = ollama.Client(host=config.OLLAMA_HOST)
        return self._client

    def _retrieve_context(self, query: str, n: int = 5) -> str:
        """Tim context tu vector memory."""
        results = self.vector.search(query, n_results=n)
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            source = meta.get("source", "")
            title = meta.get("title", "")
            header = f"[{i}] {title} ({source})" if title else f"[{i}] {source}"
            parts.append(f"{header}\n{r['text'][:600]}")
        return "\n\n".join(parts)

    def ask(self, query: str) -> dict:
        """Hoi va tra loi — RAG pipeline."""
        context = self._retrieve_context(query)
        user_msg = query
        if context:
            user_msg = f"CONTEXT TU KNOWLEDGE BASE:\n{context}\n\nCAU HOI: {query}"

        history = self.structured.get_chat_history(limit=10)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_msg})

        try:
            client = self._get_client()
            resp = client.chat(
                model=config.LLM_MODEL,
                messages=messages,
                think=False,
                options={"num_predict": 1000, "temperature": 0.3},
            )
            text = (resp["message"].get("content") or "").strip()

            self.structured.save_chat("user", query)
            self.structured.save_chat("assistant", text)

            action = self._check_action(text)
            if action:
                return {"text": text, "action": action}

            return {"text": text, "action": None}
        except Exception as e:
            return {"text": f"Loi LLM: {e}", "action": None}

    def _check_action(self, text: str) -> dict | None:
        """Kiem tra neu LLM tra ve action JSON."""
        m = re.search(r'\{[^{}]*"action"[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None

    def search_knowledge(self, query: str, n: int = 5) -> list[dict]:
        """Tim kiem truc tiep tu vector store."""
        return self.vector.search(query, n_results=n)

    def get_stats(self) -> dict:
        """Thong ke knowledge base."""
        return {
            "vector_count": self.vector.count(),
            "categories": self.vector.get_categories(),
            "structured": self.structured.stats(),
        }
