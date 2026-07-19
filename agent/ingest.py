"""Ingest pipeline — parse, chunk, embed, and store documents."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from . import config
from .memory import VectorMemory


def _chunk_text(text: str, chunk_size: int = 0, overlap: int = 0) -> list[str]:
    """Chia text thanh cac chunk nho."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > chunk_size:
            if current:
                chunks.append(current.strip())
            current = para
            if len(current) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", current)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 > chunk_size:
                        if current:
                            chunks.append(current.strip())
                        current = sent
                    else:
                        current = (current + " " + sent).strip()
        else:
            current = (current + "\n\n" + para).strip()

    if current:
        chunks.append(current.strip())

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + " " + chunks[i])
        chunks = overlapped

    return [c for c in chunks if c]


def _doc_id(text: str, source: str) -> str:
    h = hashlib.md5(f"{source}:{text[:200]}".encode()).hexdigest()[:12]
    return f"{source.replace('/', '_')}_{h}"


def parse_markdown(content: str, source: str) -> list[dict]:
    """Parse markdown thanh chunks voi metadata."""
    sections = re.split(r"\n(?=#{1,3}\s)", content)
    docs = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        title_match = re.match(r"^#{1,3}\s+(.+)", section)
        title = title_match.group(1) if title_match else ""
        chunks = _chunk_text(section)
        for chunk in chunks:
            docs.append({
                "id": _doc_id(chunk, source),
                "text": chunk,
                "metadata": {
                    "source": source,
                    "title": title,
                    "category": _guess_category(source),
                },
            })
    return docs


def parse_html(content: str, source: str) -> list[dict]:
    """Parse HTML thanh chunks."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(content, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return parse_markdown(text, source)


def parse_text(content: str, source: str) -> list[dict]:
    """Parse plain text thanh chunks."""
    chunks = _chunk_text(content)
    docs = []
    for chunk in chunks:
        docs.append({
            "id": _doc_id(chunk, source),
            "text": chunk,
            "metadata": {
                "source": source,
                "title": "",
                "category": _guess_category(source),
            },
        })
    return docs


def _guess_category(source: str) -> str:
    s = source.lower()
    for cat, info in config.TECH_CATEGORIES.items():
        if cat in s or info["name"].lower() in s:
            return cat
    return "general"


def ingest_file(filepath: Path, vector_mem: VectorMemory) -> int:
    """Doc file, parse, chunk, embed, luu vao vector store. So luong chunk da luu."""
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    suffix = filepath.suffix.lower()
    source = str(filepath.relative_to(config.DOCS_DIR)) if filepath.is_relative_to(config.DOCS_DIR) else str(filepath)

    if suffix in (".md", ".markdown"):
        docs = parse_markdown(content, source)
    elif suffix in (".html", ".htm"):
        docs = parse_html(content, source)
    else:
        docs = parse_text(content, source)

    if not docs:
        return 0

    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metas = [d["metadata"] for d in docs]
    vector_mem.add_batch(ids, texts, metas)
    return len(docs)


def ingest_directory(dirpath: Path, vector_mem: VectorMemory) -> int:
    """Ingest tat ca file trong thu muc."""
    total = 0
    for f in dirpath.rglob("*"):
        if f.is_file() and f.suffix in (".md", ".txt", ".html", ".htm", ".rst"):
            count = ingest_file(f, vector_mem)
            if count:
                print(f"  [ingest] {f.name}: {count} chunks")
            total += count
    return total


def ingest_text(text: str, source: str, category: str, vector_mem: VectorMemory) -> int:
    """Ingest text truc tiep (tu user input hoac web scraping)."""
    docs = parse_text(text, source)
    for d in docs:
        d["metadata"]["category"] = category
    if not docs:
        return 0
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metas = [d["metadata"] for d in docs]
    vector_mem.add_batch(ids, texts, metas)
    return len(docs)
