"""Research pipeline — phan tich sau, trich dan, tong hop nghien cuu."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime

import ollama

from . import config
from .memory import VectorMemory, StructuredMemory

RESEARCH_SYSTEM = """Ban la Research AI Agent — chuyen gia nghien cuu ky thuat.
Ban co truy cap offline knowledge base va co the phan tich sau, tong hop, trich dan chinh xac.

NGUYEN TAC NGHIEN CUU:
1. LUON trich dan nguon — neu thong tin tu KB, phai ghi [source: ...]
2. Phan tich nhieu khia canh — khong chi tra loi 1 phia
3. Liet ke nguon tham khao cu the (ten doc, URL neu co)
4. Phan biet ro "tu KB" vs "tu kinh nghiem AI"
5. Neu thieu thong tin noi ro — KHONG DU DOAN
6. Dinh dang bao cao: heading, bullet, code block, table
7. Tong hop thanh "Research Summary" cuoi cung

DINH DANG BA CAO:
## [Tieu de nghien cuu]
### Tom tat (Executive Summary)
### Phan tich chi tiet
### Nguon tham khao
### Ket luan va de xuat
"""

ANALYSIS_SYSTEM = """Ban la Data Analyst — phan tich du lieu ky thuat.
Dua vao du lieu tu knowledge base, hay:
1. Phan tich xu huong, pattern, correlation
2. So sanh cac luong trinh/ky thuat
3. Danh gia pros/cons chi tiet
4. Du doan xu huong tuong lai (neu co du kien)
5. De xuat best practices
"""


class ResearchAgent:
    """Nghien cuu sau — multi-step research pipeline."""

    def __init__(self, vector_mem: VectorMemory, struct_mem: StructuredMemory):
        self.vector = vector_mem
        self.structured = struct_mem
        self._client = None

    def _get_client(self) -> ollama.Client:
        if self._client is None:
            self._client = ollama.Client(host=config.OLLAMA_HOST)
        return self._client

    def _retrieve_multi(self, query: str, n: int = 12) -> list[dict]:
        """Lay nhieu context tu nhieu goc do khac nhau."""
        results = self.vector.search(query, n_results=n)
        return results

    def _build_citations(self, results: list[dict]) -> tuple[str, list[str]]:
        """Xay dung citations va context string."""
        citations = []
        parts = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            source = meta.get("source", "")
            title = meta.get("title", "")
            cite_id = f"[{i}]"
            cite_str = f"{cite_id} {title} ({source})" if title else f"{cite_id} {source}"
            citations.append(cite_str)
            parts.append(f"{cite_id} {r['text'][:800]}")
        return "\n\n".join(parts), citations

    def research(self, topic: str, depth: str = "full") -> dict:
        """Nghien cuu sau ve 1 chu de.
        depth: 'quick' | 'full' | 'deep'
        """
        n = {"quick": 6, "full": 10, "deep": 15}.get(depth, 10)
        results = self._retrieve_multi(topic, n=n)
        context, citations = self._build_citations(results)

        if not context:
            return {
                "title": topic,
                "summary": f"Khong tim thay thong tin trong KB ve: {topic}",
                "body": "",
                "citations": [],
                "timestamp": datetime.now().isoformat(),
            }

        user_msg = f"""NGUON TU KNOWLEDGE BASE:
{context}

---
CHU DE NGHIEN CUU: {topic}

Hay viet bao cao nghien cuu day du theo dinh dang:
## [Tieu de]
### Tom tat
### Phan tich chi tiet (co trich dan [source])
### So sanh (neu co)
### Best practices
### Nguon tham khao (liet ke tat ca [1], [2]...)
### Ket luan

QUAN TRONG: Moi thong tin phai co trich dan [source]. Viet bang tieng Viet."""

        messages = [
            {"role": "system", "content": RESEARCH_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        try:
            client = self._get_client()
            resp = client.chat(
                model=config.LLM_MODEL,
                messages=messages,
                think=False,
                options={
                    "num_predict": 8192 if depth == "deep" else 4096,
                    "temperature": 0.2,
                    "top_p": 0.85,
                },
            )
            text = (resp["message"].get("content") or "").strip()

            report = {
                "title": topic,
                "body": text,
                "citations": citations,
                "sources_used": len(results),
                "depth": depth,
                "timestamp": datetime.now().isoformat(),
            }

            self.structured.add_note(
                title=f"Nghien cuu: {topic}",
                content=text,
                category="research",
                tags=f"research,{depth}",
            )

            return report
        except Exception as e:
            return {"title": topic, "summary": f"Loi: {e}", "body": "", "citations": [], "timestamp": datetime.now().isoformat()}

    def analyze(self, question: str) -> dict:
        """Phan tich ky thuat — comparison, evaluation, recommendation."""
        results = self._retrieve_multi(question, n=8)
        context, citations = self._build_citations(results)

        user_msg = f"""DU LIEU:
{context}

---
CAU HOI PHAN TICH: {question}

Hay phan tich chi tiet, so sanh, danh gia. Dinh dang:
### Phan tich
### So sanh (bang/table)
### Danh gia pros/cons
### De xuat
### Nguon"""

        messages = [
            {"role": "system", "content": ANALYSIS_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        try:
            client = self._get_client()
            resp = client.chat(
                model=config.LLM_MODEL,
                messages=messages,
                think=False,
                options={"num_predict": 4096, "temperature": 0.2},
            )
            text = (resp["message"].get("content") or "").strip()
            return {"body": text, "citations": citations, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            return {"body": f"Loi: {e}", "citations": [], "timestamp": datetime.now().isoformat()}

    def summarize_topic(self, topic: str) -> str:
        """Tom tat 1 chu de tu KB."""
        results = self._retrieve_multi(topic, n=5)
        if not results:
            return f"Khong tim thay thong tin ve: {topic}"
        context, _ = self._build_citations(results)
        user_msg = f"CONTEXT:\n{context}\n\nHay tom tat ve: {topic} (200-300 tu, bang tieng Viet)"

        try:
            client = self._get_client()
            resp = client.chat(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Tom tat ngan gon, chinh xac. Bang tieng Viet."},
                    {"role": "user", "content": user_msg},
                ],
                think=False,
                options={"num_predict": 500, "temperature": 0.2},
            )
            return (resp["message"].get("content") or "").strip()
        except Exception as e:
            return f"Loi: {e}"

    def get_research_history(self, limit: int = 10) -> list[dict]:
        """Lay lich su nghien cuu da luu."""
        return self.structured.search_notes("nghien cuu", limit=limit)
