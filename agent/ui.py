"""Web UI — Gradio interface cho Research Library."""
from __future__ import annotations

import time

import gradio as gr

from . import config
from .memory import VectorMemory, StructuredMemory
from .brain import Brain
from .research import ResearchAgent
from .library import Library
from .export import (
    export_markdown,
    export_research_history,
    export_library_report,
    export_json,
)
from .graph import KnowledgeGraph


def init_components():
    """Khoi tao tat ca components."""
    vector_mem = VectorMemory(config.CHROMA_DIR)
    struct_mem = StructuredMemory(config.DB_PATH)
    brain = Brain(vector_mem)
    research = ResearchAgent(vector_mem, struct_mem)
    library = Library(vector_mem, struct_mem)
    graph = KnowledgeGraph(vector_mem)
    return vector_mem, struct_mem, brain, research, library, graph


COMPONENTS = None


def get_components():
    global COMPONENTS
    if COMPONENTS is None:
        COMPONENTS = init_components()
    return COMPONENTS


# --- Chat functions ---

def chat_respond(message: str, history: list) -> str:
    """Chat voi brain agent."""
    _, _, brain, _, _, _ = get_components()
    try:
        answer = brain.ask(message, deep=True)
        return answer
    except Exception as e:
        return f"Loi: {e}"


def research_respond(message: str, depth: str) -> str:
    """Nghien cuu sau."""
    _, _, _, research, _, _ = get_components()
    report = research.research(message, depth=depth)
    body = report.get("body", "")
    citations = report.get("citations", [])
    if citations:
        body += "\n\n---\n**Nguon:**\n" + "\n".join(f"- {c}" for c in citations)
    return body


def analyze_respond(message: str) -> str:
    """Phan tich ky thuat."""
    _, _, _, research, _, _ = get_components()
    result = research.analyze(message)
    body = result.get("body", "")
    citations = result.get("citations", [])
    if citations:
        body += "\n\n---\n**Nguon:**\n" + "\n".join(f"- {c}" for c in citations)
    return body


# --- Library functions ---

def library_overview() -> str:
    _, _, _, _, library, _ = get_components()
    overview = library.get_overview()
    sources = library.get_top_sources()
    lines = [
        f"# Library Overview",
        f"",
        f"- **Tong documents:** {overview['total_documents']}",
        f"- **Snippets:** {overview['snippets']}",
        f"- **Notes:** {overview['notes']}",
        f"",
        f"## Categories",
    ]
    for cat, count in overview.get("categories", {}).items():
        lines.append(f"- **{cat}:** {count} docs")
    lines.append(f"\n## Nguon nhieu nhat")
    for s in sources[:10]:
        lines.append(f"- [{s['category']}] {s['source']}: {s['count']} chunks")
    return "\n".join(lines)


def library_search(query: str, category: str) -> str:
    _, _, _, _, library, _ = get_components()
    cat = category.strip() if category.strip() else None
    results = library.search_full(query, n=10, category=cat)
    if not results:
        return "Khong tim thay ket qua."
    lines = [f"# Tim kiem: {query}\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. [{r['category']}] {r['title']}")
        lines.append(f"**Source:** `{r['source']}`")
        lines.append(f"**Score:** {r['distance']:.3f}")
        lines.append(f"```\n{r['text'][:300]}\n```\n")
    return "\n".join(lines)


def browse_category(category: str) -> str:
    _, _, _, _, library, _ = get_components()
    results = library.browse_category(category.strip() or "all", limit=15)
    if not results:
        return "Khong co document nao."
    lines = [f"# Category: {category}\n"]
    for r in results:
        lines.append(f"- **{r['title']}** ({r['source']})")
        lines.append(f"  {r['text'][:120]}...")
    return "\n".join(lines)


# --- Graph functions ---

def show_graph() -> str:
    _, _, _, _, _, graph = get_components()
    md = graph.export_graph_markdown()
    return md


def related_topics(topic: str) -> str:
    _, _, _, _, _, graph = get_components()
    related = graph.find_related_topics(topic)
    if not related:
        return "Khong tim thay topic lien quan."
    lines = [f"# Topic lien quan: {topic}\n"]
    for r in related:
        lines.append(f"## {r['category'].title()} (relevance: {r['relevance']:.2f})")
        lines.append(f"- Sources: {', '.join(r['sources'][:5])}")
        lines.append(f"- Sample titles:")
        for t in r["sample_titles"]:
            lines.append(f"  - {t}")
        lines.append("")
    return "\n".join(lines)


# --- Export functions ---

def export_report(topic: str, depth: str) -> str:
    _, _, _, research, _, _ = get_components()
    report = research.research(topic, depth=depth)
    filepath = export_markdown(report)
    return f"Da xuat thanh cong!\n{filepath}"


def export_full_history() -> str:
    _, _, struct_mem, _, _, _ = get_components()
    filepath = export_research_history(struct_mem)
    return f"Da xuat lich su nghien cuu!\n{filepath}"


def export_lib_report() -> str:
    _, _, _, _, library, _ = get_components()
    filepath = export_library_report(library)
    return f"Da xuat bao cao library!\n{filepath}"


def export_knowledge_graph() -> str:
    _, _, _, _, _, graph = get_components()
    filepath = export_graph_markdown(graph)
    return f"Da xuat knowledge graph!\n{filepath}"


# --- Build UI ---

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Tech Knowledge Agent") as app:
        gr.Markdown(
            """
            <div class="header">
            <h1>📚 AI Tech Knowledge Agent</h1>
            <p>Offline Research Library — Nha khoa học ky thuat AI</p>
            </div>
            """,
        )

        with gr.Tabs():
            # === TAB: Chat ===
            with gr.Tab("💬 Chat", id="chat"):
                chatbot = gr.ChatInterface(
                    fn=chat_respond,
                    title="Chat voi AI Agent",
                    description="Hoi bat cu dieu gi ve ky thuat. Agent dung KB offline de tra loi.",
                    examples=[
                        "Python decorator la gi? Cho vi du",
                        "So sanh Go va Rust cho backend",
                        "Git merge vs rebase khi nao dung?",
                    ],
                )

            # === TAB: Research ===
            with gr.Tab("🔬 Research", id="research"):
                gr.Markdown("### Nghien cuu sau — Phan tich tong hop voi trich dan")
                with gr.Row():
                    research_input = gr.Textbox(
                        label="Chu de nghien cuu",
                        placeholder="VD: So sanh message queue systems...",
                        scale=3,
                    )
                    research_depth = gr.Radio(
                        ["quick", "full", "deep"],
                        value="full",
                        label="Do sau",
                        scale=1,
                    )
                research_btn = gr.Button("🔬 Bat dau nghien cuu", variant="primary")
                research_output = gr.Markdown(label="Ket qua nghien cuu")

                research_btn.click(
                    fn=research_respond,
                    inputs=[research_input, research_depth],
                    outputs=research_output,
                )

                gr.Markdown("---")
                gr.Markdown("### Phan tich ky thuat")
                analyze_input = gr.Textbox(
                    label="Cau hoi phan tich",
                    placeholder="VD: Docker vs Kubernetes cho microservices?",
                )
                analyze_btn = gr.Button("📊 Phan tich")
                analyze_output = gr.Markdown(label="Ket qua phan tich")
                analyze_btn.click(
                    fn=analyze_respond,
                    inputs=analyze_input,
                    outputs=analyze_output,
                )

            # === TAB: Library ===
            with gr.Tab("📖 Library", id="library"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Tong quan")
                        lib_overview_btn = gr.Button("Xem tong quan")
                        lib_overview_output = gr.Markdown()
                        lib_overview_btn.click(fn=library_overview, outputs=lib_overview_output)

                    with gr.Column(scale=2):
                        gr.Markdown("### Tim kiem")
                        with gr.Row():
                            search_input = gr.Textbox(label="Query", scale=3)
                            search_cat = gr.Textbox(label="Category", scale=1, placeholder="python, go...")
                        search_btn = gr.Button("Tim kiem")
                        search_output = gr.Markdown()
                        search_btn.click(
                            fn=library_search,
                            inputs=[search_input, search_cat],
                            outputs=search_output,
                        )

                gr.Markdown("---")
                gr.Markdown("### Duyet theo category")
                cat_input = gr.Textbox(label="Category", placeholder="python, linux, go, rust, git...")
                browse_btn = gr.Button("Duyet")
                browse_output = gr.Markdown()
                browse_btn.click(fn=browse_category, inputs=cat_input, outputs=browse_output)

            # === TAB: Knowledge Graph ===
            with gr.Tab("🕸️ Graph", id="graph"):
                gr.Markdown("### Knowledge Graph — Moi quan he giua topics")
                graph_btn = gr.Button("Xem graph")
                graph_output = gr.Markdown()
                graph_btn.click(fn=show_graph, outputs=graph_output)

                gr.Markdown("---")
                gr.Markdown("### Tim topic lien quan")
                topic_input = gr.Textbox(label="Topic")
                related_btn = gr.Button("Tim lien quan")
                related_output = gr.Markdown()
                related_btn.click(fn=related_topics, inputs=topic_input, outputs=related_output)

            # === TAB: Export ===
            with gr.Tab("📤 Export", id="export"):
                gr.Markdown("### Xuat bao cao nghien cuu")
                with gr.Row():
                    export_topic = gr.Textbox(label="Chu de", placeholder="Python concurrency", scale=3)
                    export_depth = gr.Radio(["quick", "full", "deep"], value="full", label="Do sau", scale=1)
                export_report_btn = gr.Button("📄 Xuat markdown")
                export_report_output = gr.Textbox(label="File path")
                export_report_btn.click(
                    fn=export_report,
                    inputs=[export_topic, export_depth],
                    outputs=export_report_output,
                )

                gr.Markdown("---")
                gr.Markdown("### Xuat toan bo")
                with gr.Row():
                    export_hist_btn = gr.Button("Xuat lich su nghien cuu")
                    export_lib_btn = gr.Button("Xuat bao cao library")
                    export_graph_btn = gr.Button("Xuat knowledge graph")
                export_all_output = gr.Textbox(label="Result")

                export_hist_btn.click(fn=export_full_history, outputs=export_all_output)
                export_lib_btn.click(fn=export_lib_report, outputs=export_all_output)
                export_graph_btn.click(fn=export_knowledge_graph, outputs=export_all_output)

            # === TAB: Stats ===
            with gr.Tab("📊 Stats", id="stats"):
                gr.Markdown("### Thong ke he thong")
                stats_btn = gr.Button("Lam moi")
                stats_output = gr.Markdown()

                def show_stats():
                    _, _, _, _, library, _ = get_components()
                    overview = library.get_overview()
                    md = f"""## Thong ke he thong

| Chi so | Gia tri |
|--------|---------|
| Tong documents | {overview['total_documents']} |
| Snippets | {overview['snippets']} |
| Notes | {overview['notes']} |
| Categories | {len(overview.get('categories', {}))} |

### Categories chi tiet
"""
                    for cat, count in overview.get("categories", {}).items():
                        md += f"- **{cat}:** {count} documents\n"

                    md += f"\n### Config\n- LLM: {config.LLM_MODEL}"
                    md += f"\n- Embedding: {config.EMBED_MODEL}"
                    md += f"\n- Chunk size: {config.CHUNK_SIZE}"
                    md += f"\n- Ollama: {config.OLLAMA_HOST}"
                    return md

                stats_btn.click(fn=show_stats, outputs=stats_output)

    return app


def launch_ui(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    """Khoi chay web UI."""
    app = build_ui()
    app.launch(server_name=host, server_port=port, share=share)
