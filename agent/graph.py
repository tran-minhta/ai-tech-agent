"""Knowledge graph — cross-reference, topic mapping, relationship discovery."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import config
from .memory import VectorMemory


class KnowledgeGraph:
    """Xay dung knowledge graph tu vector embeddings."""

    def __init__(self, vector_mem: VectorMemory):
        self.vector = vector_mem
        self._cache_path = config.DATA_DIR / "knowledge_graph.json"

    def build_category_graph(self) -> dict:
        """Xay dung graph relationships giua categories."""
        all_data = self.vector.get(include=["metadatas", "documents"])
        if not all_data["ids"]:
            return {"nodes": [], "edges": [], "stats": {}}

        category_chunks: dict[str, list] = defaultdict(list)
        for i, meta in enumerate(all_data["metadatas"]):
            cat = meta.get("category", "unknown")
            doc = all_data["documents"][i] if i < len(all_data["documents"]) else ""
            category_chunks[cat].append({
                "id": all_data["ids"][i],
                "source": meta.get("source", ""),
                "title": meta.get("title", ""),
                "text_preview": doc[:150],
            })

        nodes = []
        for cat, chunks in category_chunks.items():
            sources = list({c["source"] for c in chunks})
            nodes.append({
                "id": cat,
                "label": cat.title(),
                "chunk_count": len(chunks),
                "source_count": len(sources),
                "sources": sources[:20],
            })

        edges = []
        seen_edges = set()
        for i, meta_i in enumerate(all_data["metadatas"]):
            cat_i = meta_i.get("category", "unknown")
            for j, meta_j in enumerate(all_data["metadatas"]):
                if j <= i:
                    continue
                cat_j = meta_j.get("category", "unknown")
                if cat_i == cat_j:
                    continue
                key = tuple(sorted([cat_i, cat_j]))
                if key in seen_edges:
                    continue
                src_i = meta_i.get("source", "")
                src_j = meta_j.get("source", "")
                if src_i and src_j and src_i == src_j:
                    edges.append({
                        "source": cat_i,
                        "target": cat_j,
                        "shared_source": src_i,
                        "weight": 1,
                    })
                    seen_edges.add(key)

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "total_chunks": len(all_data["ids"]),
            },
        }

    def find_related_topics(self, topic: str, n: int = 8) -> list[dict]:
        """Tim cac topic lien quan den 1 chu de."""
        results = self.vector.search(topic, n_results=n)
        related: dict[str, dict] = {}
        for r in results:
            cat = r["metadata"].get("category", "unknown")
            if cat not in related:
                related[cat] = {
                    "category": cat,
                    "relevance": 0,
                    "sources": set(),
                    "sample_titles": [],
                }
            related[cat]["relevance"] += 1 / (r["distance"] + 0.01)
            src = r["metadata"].get("source", "")
            if src:
                related[cat]["sources"].add(src)
            title = r["metadata"].get("title", "")
            if title and title not in related[cat]["sample_titles"]:
                related[cat]["sample_titles"].append(title)

        out = []
        for cat, info in related.items():
            info["sources"] = list(info["sources"])
            info["sample_titles"] = info["sample_titles"][:5]
            out.append(info)
        out.sort(key=lambda x: x["relevance"], reverse=True)
        return out

    def source_network(self, source_url: str) -> dict:
        """Phat hien mang luoi cua 1 nguon — cau truc, chu de con."""
        results = self.vector.search(source_url, n_results=30, where={"source": source_url})
        if not results:
            return {"source": source_url, "subtopics": [], "total_chunks": 0}

        subtopics: dict[str, list] = defaultdict(list)
        for r in results:
            cat = r["metadata"].get("category", "unknown")
            subtopics[cat].append(r["text"][:100])

        return {
            "source": source_url,
            "total_chunks": len(results),
            "subtopics": {k: {"count": len(v), "samples": v[:3]} for k, v in subtopics.items()},
        }

    def export_graph_json(self, output_path: Path | None = None) -> Path:
        """Xuat graph thanh JSON de hien thi."""
        graph = self.build_category_graph()
        out = output_path or (config.DATA_DIR / "knowledge_graph.json")
        out.write_text(json.dumps(graph, indent=2, ensure_ascii=False))
        return out

    def export_graph_mermaid(self) -> str:
        """Xuat graph thanh Mermaid diagram syntax."""
        graph = self.build_category_graph()
        lines = ["graph LR"]
        for node in graph["nodes"]:
            safe_id = node["id"].replace("-", "_").replace(" ", "_")
            lines.append(f"    {safe_id}[\"{node['label']}<br/>{node['chunk_count']} chunks\"]")
        for edge in graph["edges"]:
            s = edge["source"].replace("-", "_").replace(" ", "_")
            t = edge["target"].replace("-", "_").replace(" ", "_")
            lines.append(f"    {s} -->|\"{edge['shared_source']}\"| {t}")
        return "\n".join(lines)

    def export_graph_markdown(self) -> str:
        """Xuat graph thanh markdown voi diagram."""
        graph = self.build_category_graph()
        mermaid = self.export_graph_mermaid()

        md = f"""# Knowledge Graph

**Tong nodes:** {graph['stats']['total_nodes']}
**Tong edges:** {graph['stats']['total_edges']}
**Tong chunks:** {graph['stats']['total_chunks']}

## Category Map

```mermaid
{mermaid}
```

## Chi tiet

"""
        for node in graph["nodes"]:
            md += f"### {node['label']}\n"
            md += f"- Chunks: {node['chunk_count']}\n"
            md += f"- Sources: {node['source_count']}\n"
            for s in node["sources"][:5]:
                md += f"  - `{s}`\n"
            md += "\n"

        if graph["edges"]:
            md += "## Relationships\n\n"
            md += "| From | To | Shared Source |\n|------|-----|---------------|\n"
            for edge in graph["edges"]:
                md += f"| {edge['source']} | {edge['target']} | {edge['shared_source']} |\n"

        return md
