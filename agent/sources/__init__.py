"""Sources — tai va index documentation offline."""
from __future__ import annotations

import time
from pathlib import Path

import requests

from .. import config
from ..ingest import ingest_text, ingest_file
from ..memory import VectorMemory


def _fetch_url(url: str, timeout: int = 30) -> str | None:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": config.APP_NAME})
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  [source] Loi tai {url}: {e}")
        return None


PYTHON_DOCS = [
    ("https://docs.python.org/3/tutorial/index.html", "python", "Python Tutorial"),
    ("https://docs.python.org/3/library/index.html", "python", "Python Standard Library"),
    ("https://docs.python.org/3/howto/functional.html", "python", "Functional Programming HOWTO"),
    ("https://docs.python.org/3/howto/regex.html", "python", "Regular Expression HOWTO"),
]

LINUX_DOCS = [
    ("https://www.kernel.org/doc/html/latest/", "linux", "Linux Kernel Documentation"),
]

SHELL_DOCS = [
    ("https://www.gnu.org/software/bash/manual/bash.html", "shell", "Bash Manual"),
]

GO_DOCS = [
    ("https://go.dev/doc/effective_go", "go", "Effective Go"),
    ("https://go.dev/doc/", "go", "Go Documentation"),
]

RUST_DOCS = [
    ("https://doc.rust-lang.org/book/", "rust", "The Rust Book"),
]

GIT_DOCS = [
    ("https://git-scm.com/book/en/v2", "git", "Pro Git Book"),
]


def _ingest_from_urls(urls: list[tuple[str, str, str]], vector_mem: VectorMemory, label: str):
    """Tai va ingest tu danh sach URLs."""
    print(f"[source] Dang tai {label}...")
    count = 0
    for url, category, title in urls:
        print(f"  -> {title} ({url})")
        html = _fetch_url(url)
        if not html:
            continue
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        n = ingest_text(text, f"{category}/{title}", category, vector_mem)
        count += n
        print(f"     {n} chunks")
        time.sleep(0.5)
    print(f"[source] {label}: {count} chunks tong cong")


def ingest_python_docs(vector_mem: VectorMemory):
    _ingest_from_urls(PYTHON_DOCS, vector_mem, "Python Docs")


def ingest_linux_docs(vector_mem: VectorMemory):
    _ingest_from_urls(LINUX_DOCS, vector_mem, "Linux Docs")


def ingest_shell_docs(vector_mem: VectorMemory):
    _ingest_from_urls(SHELL_DOCS, vector_mem, "Shell/Bash Docs")


def ingest_go_docs(vector_mem: VectorMemory):
    _ingest_from_urls(GO_DOCS, vector_mem, "Go Docs")


def ingest_rust_docs(vector_mem: VectorMemory):
    _ingest_from_urls(RUST_DOCS, vector_mem, "Rust Docs")


def ingest_git_docs(vector_mem: VectorMemory):
    _ingest_from_urls(GIT_DOCS, vector_mem, "Git Docs")


def ingest_local_docs(vector_mem: VectorMemory):
    """Ingest tat ca file trong data/docs/."""
    docs_dir = config.DOCS_DIR
    if not docs_dir.exists():
        return
    count = ingest_file(docs_dir, vector_mem)
    if count:
        print(f"[source] Local docs: {count} chunks")


def ingest_all(vector_mem: VectorMemory, categories: list[str] | None = None):
    """Ingest tat ca sources."""
    all_sources = {
        "python": ingest_python_docs,
        "linux": ingest_linux_docs,
        "shell": ingest_shell_docs,
        "go": ingest_go_docs,
        "rust": ingest_rust_docs,
        "git": ingest_git_docs,
    }

    ingest_local_docs(vector_mem)

    targets = categories or list(all_sources.keys())
    for cat in targets:
        if cat in all_sources:
            try:
                all_sources[cat](vector_mem)
            except Exception as e:
                print(f"[source] Loi ingest {cat}: {e}")


def ingest_url(url: str, category: str, vector_mem: VectorMemory) -> int:
    """Tai 1 URL va ingest."""
    html = _fetch_url(url)
    if not html:
        return 0
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return ingest_text(text, url, category, vector_mem)
