"""Export layer — xuat research reports thanh markdown/PDF."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from . import config


def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    return name[:60] or "report"


def export_markdown(report: dict, output_dir: Path | None = None) -> Path:
    """Xuat research report thanh file markdown."""
    out_dir = output_dir or (config.DATA_DIR / "exports")
    out_dir.mkdir(parents=True, exist_ok=True)

    title = report.get("title", "Report")
    body = report.get("body", "")
    citations = report.get("citations", [])
    timestamp = report.get("timestamp", datetime.now().isoformat())
    depth = report.get("depth", "full")

    md = f"""# {title}

**Ngay xuat:** {timestamp}
**Do sau:** {depth}
**Nguon su dung:** {report.get('sources_used', 0)}

---

{body}

---

## Nguon tham khao
"""
    for cite in citations:
        md += f"- {cite}\n"

    md += f"\n---\n*Exported by AI Tech Knowledge Agent — {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"

    filename = f"{_sanitize_filename(title)}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    filepath = out_dir / filename
    filepath.write_text(md, encoding="utf-8")
    return filepath


def export_research_history(struct_mem, output_dir: Path | None = None) -> Path:
    """Xuat toan bo lich su nghien cuu thanh 1 file markdown."""
    out_dir = output_dir or (config.DATA_DIR / "exports")
    out_dir.mkdir(parents=True, exist_ok=True)

    notes = struct_mem.search_notes("nghien cuu", limit=50)
    md = f"""# Research History

**Ngay xuat:** {datetime.now().isoformat()}
**Tong so bao cao:** {len(notes)}

---

"""
    for i, note in enumerate(notes, 1):
        md += f"""## {i}. {note.get('title', 'Untitled')}
*{datetime.fromtimestamp(note.get('created_at', 0)).strftime('%Y-%m-%d %H:%M')}*

{note.get('content', '')}

---

"""

    filename = f"research_history_{datetime.now().strftime('%Y%m%d')}.md"
    filepath = out_dir / filename
    filepath.write_text(md, encoding="utf-8")
    return filepath


def export_library_report(library, output_dir: Path | None = None) -> Path:
    """Xuat tong quan library thanh markdown."""
    out_dir = output_dir or (config.DATA_DIR / "exports")
    out_dir.mkdir(parents=True, exist_ok=True)

    overview = library.get_overview()
    sources = library.get_top_sources()
    collections = library.get_collections()

    md = f"""# Library Report

**Ngay xuat:** {datetime.now().isoformat()}

## Tong quan
- **Tong documents:** {overview['total_documents']}
- **Snippets:** {overview['snippets']}
- **Notes:** {overview['notes']}

## Categories
"""
    for cat, count in overview.get("categories", {}).items():
        md += f"- **{cat}:** {count} documents\n"

    md += "\n## Nguon nhieu nhat\n"
    for s in sources[:15]:
        md += f"- [{s['category']}] {s['source']}: {s['count']} chunks\n"

    if collections:
        md += "\n## Bo suu tap\n"
        for name, info in collections.items():
            md += f"- **{name}:** {info.get('description', '')} ({len(info.get('topic_ids', []))} topics)\n"

    md += f"\n---\n*Exported by AI Tech Knowledge Agent — {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"

    filename = f"library_report_{datetime.now().strftime('%Y%m%d')}.md"
    filepath = out_dir / filename
    filepath.write_text(md, encoding="utf-8")
    return filepath


def export_json(data: dict, name: str, output_dir: Path | None = None) -> Path:
    """Xuat du lieu thanh JSON."""
    out_dir = output_dir or (config.DATA_DIR / "exports")
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{_sanitize_filename(name)}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    filepath = out_dir / filename
    filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return filepath
