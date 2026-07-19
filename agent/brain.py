"""Brain layer — RAG query pipeline + LLM response generation (deep mode)."""
from __future__ import annotations

import json
import re

import ollama

from . import config
from .memory import VectorMemory, StructuredMemory

SYSTEM_PROMPT = """Ban la AI Tech Knowledge Agent — mot chuyen gia lap trinh, OS, va cong nghe cap cao.
Ban co long-term memory luu tru knowledge va co the tra cuu offline.

NGUYEN TAC TRA LOI:
1. Su dung context tu knowledge base de tra loi CHINH XAC va CHI TIET
2. Tra loi DAY DU, sau sac — khong duoc qua ngan gon
3. Luon cho vi du code THUC TE, chay duoc khi hoi ve lap trinh
4. Giai thich CA TAI SAO va CA LAM THE NAO — khong chi noi la gi
5. Neu co nhieu cach giai quyet, liet ke TAT CA va so sanh
6. Neu co context tu KB, TAN DUNG TOAN BO — khong bo qua thong tin
7. Tra loi bang tieng Viet neu nguoi dung hoi bang tieng Viet
8. Neu khong biet, noi ro "Khong co thong tin trong KB" — KHONG DU DOAN
9. De trong dong de doc, su dung headings va bullet points

KHI TRA LOI VE LAP TRINH, PHAI CO:
- Gioi thieu ngan gon (1-2 dong)
- Giai thich chi tiet thuat toan/co che
- Vi du code day du, chay duoc
- So sanh voi cac cach khac (neu co)
- Best practices va common pitfalls
- Tai lieu tham khao (neu co trong KB)

Neu nguoi dung muon luu note/snippet, tra ve JSON:
{"action": "save_note", "title": "...", "content": "...", "category": "..."}
{"action": "save_snippet", "title": "...", "code": "...", "language": "...", "category": "..."}
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

    def _retrieve_context(self, query: str, n: int = 8) -> str:
        """Tim nhieu context tu vector memory."""
        results = self.vector.search(query, n_results=n)
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            source = meta.get("source", "")
            title = meta.get("title", "")
            header = f"[{i}] {title} ({source})" if title else f"[{i}] {source}"
            parts.append(f"{header}\n{r['text'][:1000]}")
        return "\n\n".join(parts)

    def ask(self, query: str, deep: bool = True) -> dict:
        """Hoi va tra loi — RAG pipeline voi deep mode."""
        n_context = 10 if deep else 5
        context = self._retrieve_context(query, n=n_context)

        user_msg = query
        if context:
            user_msg = f"CONTEXT TU KNOWLEDGE BASE (su dung tat ca de tra loi chi tiet):\n\n{context}\n\n---\nCAU HOI: {query}\n\nHay tra loi DAY DU, sau sac, voi vi du code thuc te."

        history = self.structured.get_chat_history(limit=8)
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
                options={
                    "num_predict": 4096 if deep else 2048,
                    "temperature": 0.3,
                    "top_p": 0.9,
                },
            )
            text = (resp["message"].get("content") or "").strip()

            self.structured.save_chat("user", query)
            self.structured.save_chat("assistant", text)

            action = self._check_action(text)
            return {"text": text, "action": action}
        except Exception as e:
            return {"text": f"Loi LLM: {e}", "action": None}

    def _check_action(self, text: str) -> dict | None:
        m = re.search(r'\{[^{}]*"action"[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None

    def search_knowledge(self, query: str, n: int = 10) -> list[dict]:
        return self.vector.search(query, n_results=n)

    def get_stats(self) -> dict:
        return {
            "vector_count": self.vector.count(),
            "categories": self.vector.get_categories(),
            "structured": self.structured.stats(),
        }
