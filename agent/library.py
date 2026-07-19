"""Library layer — quan ly offline knowledge library."""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import config
from .memory import VectorMemory, StructuredMemory


class Library:
    """Offline knowledge library — browse, search, organize."""

    def __init__(self, vector_mem: VectorMemory, struct_mem: StructuredMemory):
        self.vector = vector_mem
        self.structured = struct_mem
        self._index_path = config.DATA_DIR / "library_index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text())
            except Exception:
                pass
        return {"topics": {}, "collections": {}, "bookmarks": []}

    def _save_index(self):
        self._index_path.write_text(json.dumps(self._index, indent=2, ensure_ascii=False))

    def get_overview(self) -> dict:
        """Tong quan library."""
        stats = self.vector.get_categories()
        total = self.vector.count()
        s_stats = self.structured.stats()
        return {
            "total_documents": total,
            "categories": stats,
            "snippets": s_stats["snippets"],
            "notes": s_stats["notes"],
            "research_reports": s_stats.get("notes", 0),
        }

    def browse_category(self, category: str, limit: int = 20) -> list[dict]:
        """Duyet theo category."""
        results = self.vector.search(
            category,
            n_results=limit,
            where={"category": category} if category != "all" else None,
        )
        return [
            {
                "id": r["id"],
                "text": r["text"][:200],
                "source": r["metadata"].get("source", ""),
                "title": r["metadata"].get("title", ""),
                "category": r["metadata"].get("category", ""),
            }
            for r in results
        ]

    def search_full(self, query: str, n: int = 20, category: str | None = None) -> list[dict]:
        """Tim kiem toan bo library."""
        where = {"category": category} if category else None
        results = self.vector.search(query, n_results=n, where=where)
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "source": r["metadata"].get("source", ""),
                "title": r["metadata"].get("title", ""),
                "category": r["metadata"].get("category", ""),
                "distance": r["distance"],
            }
            for r in results
        ]

    def get_top_sources(self, limit: int = 10) -> list[dict]:
        """Nguon nhieu nhat trong library."""
        if self.vector.count() == 0:
            return []
        all_data = self.vector.get(include=["metadatas"])
        source_counts: dict[str, dict] = {}
        for meta in all_data["metadatas"]:
            src = meta.get("source", "unknown")
            cat = meta.get("category", "unknown")
            if src not in source_counts:
                source_counts[src] = {"source": src, "category": cat, "count": 0}
            source_counts[src]["count"] += 1
        sorted_sources = sorted(source_counts.values(), key=lambda x: x["count"], reverse=True)
        return sorted_sources[:limit]

    def add_bookmark(self, url: str, title: str, description: str = "", category: str = ""):
        self._index["bookmarks"].append({
            "url": url,
            "title": title,
            "description": description,
            "category": category,
            "timestamp": time.time(),
        })
        self._save_index()

    def get_bookmarks(self) -> list[dict]:
        return self._index.get("bookmarks", [])

    def create_collection(self, name: str, description: str = "", topic_ids: list[str] | None = None):
        """Tao bo suu tap tu topic_ids."""
        self._index["collections"][name] = {
            "description": description,
            "topic_ids": topic_ids or [],
            "created_at": time.time(),
        }
        self._save_index()

    def get_collections(self) -> dict:
        return self._index.get("collections", {})

    def topic_links(self, topic: str, n: int = 5) -> list[dict]:
        """Tim cac topic lien quan."""
        results = self.vector.search(topic, n_results=n)
        related = set()
        for r in results:
            cat = r["metadata"].get("category", "")
            if cat and cat not in related:
                related.add(cat)
        return [{"topic": t, "count": sum(1 for r in results if r["metadata"].get("category") == t)} for t in related if t]

    def export_library_index(self) -> str:
        """Xuat library index thanh JSON string."""
        overview = self.get_overview()
        sources = self.get_top_sources()
        collections = self.get_collections()
        bookmarks = self.get_bookmarks()
        return json.dumps({
            "overview": overview,
            "top_sources": sources,
            "collections": collections,
            "bookmarks": bookmarks,
        }, indent=2, ensure_ascii=False)
