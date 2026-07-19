"""AI Tech Knowledge Agent — Offline Research Library.

Cach dung:
    python main.py chat                    # CLI chat mode
    python main.py chat --voice            # Voice + CLI chat
    python main.py web                     # Mo web UI (Gradio)
    python main.py web --port 7861         # Custom port
    python main.py research "topic"        # Nghien cuu sau
    python main.py analyze "question"      # Phan tich ky thuat
    python main.py graph                   # Xem knowledge graph
    python main.py export                  # Xuat bao cao
    python main.py ingest                  # Index tat ca docs
    python main.py ingest --category python  # Chi index Python docs
    python main.py search "python decorator" # Tim kiem truc tiep
    python main.py stats                   # Xem thong ke knowledge base
"""
import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from agent import sources
from agent.brain import Brain
from agent.ingest import ingest_text
from agent.memory import VectorMemory, StructuredMemory
from agent.research import ResearchAgent
from agent.library import Library
from agent.graph import KnowledgeGraph
from agent.export import (
    export_markdown,
    export_research_history,
    export_library_report,
)
from agent.voice import Listener, Speaker

console = Console()


def cmd_chat(args):
    """CLI chat mode voi optional voice."""
    vector_mem = VectorMemory()
    struct_mem = StructuredMemory()
    brain = Brain(vector_mem, struct_mem)
    deep = not args.shallow

    speaker = None
    listener = None
    if args.voice:
        speaker = Speaker()
        listener = Listener()
        speaker.say("AI Tech Knowledge Agent san sang. Hay hoi bat cu dieu gi!")

    console.print(Panel(
        "[bold green]AI Tech Knowledge Agent[/]\n"
        "Offline Research Library\n"
        "Lenh: [cyan]status[/] | [cyan]search <query>[/] | [cyan]save <note>[/] | [cyan]quit[/]",
        title="Ready",
    ))

    while True:
        try:
            if listener:
                text = listener.listen(prompt="\n🎤 Noi lenh:")
            else:
                text = input("\n> ").strip()

            if not text:
                continue

            text_lower = text.lower()

            if text_lower in ("quit", "exit", "thoat", "dung", "stop"):
                if speaker:
                    speaker.say("Tam biet!")
                break

            if text_lower == "status":
                _show_stats(brain, speaker)
                continue

            if text_lower.startswith("search "):
                query = text[7:].strip()
                _search(brain, query, speaker)
                continue

            if text_lower.startswith("save "):
                _save_note(struct_mem, text[5:].strip(), speaker)
                continue

            result = brain.ask(text, deep=deep)
            _display_response(result, speaker)

        except KeyboardInterrupt:
            if speaker:
                speaker.say("Tam biet!")
            break
        except Exception as e:
            console.print(f"[red]Loi: {e}[/]")


def _display_response(result: dict, speaker: Speaker | None = None):
    text = result["text"]
    action = result.get("action")
    if action:
        _handle_action(action, speaker)
    md = Markdown(text)
    console.print(Panel(md, title="[bold cyan]AI Response[/]", border_style="cyan"))


def _handle_action(action: dict, speaker: Speaker | None = None):
    act = action.get("action", "")
    if act == "save_note":
        console.print("[green]Da luu note.[/]")
        if speaker:
            speaker.say("Da luu note.")
    elif act == "save_snippet":
        console.print("[green]Da luu snippet.[/]")
        if speaker:
            speaker.say("Da luu snippet.")


def _show_stats(brain: Brain, speaker: Speaker | None = None):
    stats = brain.get_stats()
    table = Table(title="Knowledge Base Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total vectors", str(stats["vector_count"]))
    table.add_row("Snippets", str(stats["structured"]["snippets"]))
    table.add_row("Notes", str(stats["structured"]["notes"]))
    table.add_row("Chat history", str(stats["structured"]["chat_history"]))
    table.add_row("", "")
    for cat, count in stats["categories"].items():
        table.add_row(f"  {cat}", str(count))
    console.print(table)
    if speaker:
        speaker.say(f"Co {stats['vector_count']} documents trong knowledge base.")


def _search(brain: Brain, query: str, speaker: Speaker | None = None):
    results = brain.search_knowledge(query, n=5)
    if not results:
        console.print("[yellow]Khong tim thay ket qua.[/]")
        return
    table = Table(title=f"Ket qua: {query}")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Source")
    table.add_column("Title")
    table.add_column("Distance", width=8)
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        table.add_row(
            str(i),
            meta.get("source", "")[:30],
            meta.get("title", "")[:30],
            f"{r['distance']:.3f}",
        )
    console.print(table)
    if speaker:
        speaker.say(f"Tim thay {len(results)} ket qua cho '{query}'.")


def _save_note(struct_mem: StructuredMemory, text: str, speaker: Speaker | None = None):
    if not text:
        console.print("[yellow]Nhap noi dung note.[/]")
        return
    note_id = struct_mem.add_note(title=text[:50], content=text, category="user")
    console.print(f"[green]Da luu note #{note_id}[/]")
    if speaker:
        speaker.say(f"Da luu note so {note_id}.")


def cmd_research(args):
    """Nghien cuu sau ve 1 chu de."""
    vector_mem = VectorMemory()
    struct_mem = StructuredMemory()
    research = ResearchAgent(vector_mem, struct_mem)

    console.print(f"[cyan]Dang nghien cuu: {args.topic} (do sau: {args.depth})...[/]")
    report = research.research(args.topic, depth=args.depth)

    body = report.get("body", "")
    citations = report.get("citations", [])

    md = Markdown(body)
    console.print(Panel(md, title=f"[bold cyan]Nghien cuu: {args.topic}[/]", border_style="cyan"))

    if citations:
        console.print("\n[bold]Nguon tham khao:[/]")
        for c in citations:
            console.print(f"  [dim]{c}[/]")

    if args.export:
        filepath = export_markdown(report)
        console.print(f"\n[green]Da xuat: {filepath}[/]")


def cmd_analyze(args):
    """Phan tich ky thuat."""
    vector_mem = VectorMemory()
    struct_mem = StructuredMemory()
    research = ResearchAgent(vector_mem, struct_mem)

    console.print(f"[cyan]Dang phan tich: {args.question}...[/]")
    result = research.analyze(args.question)

    body = result.get("body", "")
    citations = result.get("citations", [])

    md = Markdown(body)
    console.print(Panel(md, title="[bold cyan]Phan tich ky thuat[/]", border_style="cyan"))

    if citations:
        console.print("\n[bold]Nguon:[/]")
        for c in citations:
            console.print(f"  [dim]{c}[/]")


def cmd_graph(args):
    """Xem knowledge graph."""
    vector_mem = VectorMemory()
    graph = KnowledgeGraph(vector_mem)

    if args.topic:
        related = graph.find_related_topics(args.topic)
        console.print(f"[bold]Topic lien quan: {args.topic}[/]\n")
        for r in related:
            console.print(f"  [cyan]{r['category'].title()}[/] (relevance: {r['relevance']:.2f})")
            for s in r["sources"][:3]:
                console.print(f"    [dim]{s}[/]")
    else:
        graph_data = graph.build_category_graph()
        console.print(Panel(
            f"Nodes: {graph_data['stats']['total_nodes']} | "
            f"Edges: {graph_data['stats']['total_edges']} | "
            f"Chunks: {graph_data['stats']['total_chunks']}",
            title="[bold]Knowledge Graph[/]",
        ))
        for node in graph_data["nodes"]:
            console.print(f"  [cyan]{node['label']}[/] — {node['chunk_count']} chunks, {node['source_count']} sources")
        if graph_data["edges"]:
            console.print("\n[bold]Relationships:[/]")
            for edge in graph_data["edges"]:
                console.print(f"  {edge['source']} <-> {edge['target']} ({edge['shared_source']})")

        if args.export_md:
            md_text = graph.export_graph_markdown()
            filepath = config.DATA_DIR / "exports" / "knowledge_graph.md"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(md_text, encoding="utf-8")
            console.print(f"\n[green]Da xuat: {filepath}[/]")


def cmd_export(args):
    """Xuat bao cao."""
    from agent import config

    vector_mem = VectorMemory()
    struct_mem = StructuredMemory()
    library = Library(vector_mem, struct_mem)

    console.print("[cyan]Dang xuat bao cao...[/]")

    if args.what == "all":
        fp1 = export_research_history(struct_mem)
        fp2 = export_library_report(library)
        console.print(f"[green]Lich su nghien cuu: {fp1}[/]")
        console.print(f"[green]Bao cao library: {fp2}[/]")
    elif args.what == "research":
        fp = export_research_history(struct_mem)
        console.print(f"[green]{fp}[/]")
    elif args.what == "library":
        fp = export_library_report(library)
        console.print(f"[green]{fp}[/]")
    elif args.what == "graph":
        graph = KnowledgeGraph(vector_mem)
        md_text = graph.export_graph_markdown()
        filepath = config.DATA_DIR / "exports" / "knowledge_graph.md"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(md_text, encoding="utf-8")
        console.print(f"[green]{filepath}[/]")


def cmd_web(args):
    """Mo web UI."""
    from agent.ui import launch_ui
    console.print(f"[cyan]Dang khoi chay Web UI tai http://{args.host}:{args.port}[/]")
    launch_ui(host=args.host, port=args.port, share=args.share)


def cmd_ingest(args):
    """Index docs vao knowledge base."""
    vector_mem = VectorMemory()

    if args.category:
        categories = [c.strip() for c in args.category.split(",")]
        console.print(f"[cyan]Dang index: {', '.join(categories)}[/]")
        sources.ingest_all(vector_mem, categories=categories)
    elif args.url:
        cat = args.url_category or "general"
        console.print(f"[cyan]Dang tai {args.url} ({cat})...[/]")
        n = sources.ingest_url(args.url, cat, vector_mem)
        console.print(f"[green]Done: {n} chunks[/]")
    elif args.file:
        from pathlib import Path
        from agent.ingest import ingest_file
        p = Path(args.file)
        console.print(f"[cyan]Dang index {p.name}...[/]")
        n = ingest_file(p, vector_mem)
        console.print(f"[green]Done: {n} chunks[/]")
    else:
        console.print("[cyan]Dang index tat ca sources...[/]")
        sources.ingest_all(vector_mem)

    console.print(f"[bold green]Tong: {vector_mem.count()} documents trong KB.[/]")


def cmd_search(args):
    """Tim kiem truc tiep."""
    vector_mem = VectorMemory()
    results = vector_mem.search(args.query, n_results=args.n)
    if not results:
        console.print("[yellow]Khong tim thay.[/]")
        return

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        console.print(Panel(
            f"{r['text'][:500]}\n\n"
            f"[dim]Source: {meta.get('source', '')} | Title: {meta.get('title', '')} | Distance: {r['distance']:.3f}[/]",
            title=f"[{i}]",
        ))


def cmd_stats(args):
    """Xem thong ke."""
    vector_mem = VectorMemory()
    struct_mem = StructuredMemory()

    table = Table(title="Knowledge Base Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total vectors", str(vector_mem.count()))
    for cat, count in vector_mem.get_categories().items():
        table.add_row(f"  {cat}", str(count))
    table.add_row("", "")
    stats = struct_mem.stats()
    table.add_row("Snippets", str(stats["snippets"]))
    table.add_row("Notes", str(stats["notes"]))
    table.add_row("Chat history", str(stats["chat_history"]))
    console.print(table)


def build_parser():
    p = argparse.ArgumentParser(
        description="AI Tech Knowledge Agent — Offline Research Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  chat      Chat voi agent (CLI)
  web       Mo web UI (Gradio)
  research  Nghien cuu sau ve 1 chu de
  analyze   Phan tich ky thuat
  graph     Xem knowledge graph
  export    Xuat bao cao
  ingest    Index docs vao knowledge base
  search    Tim kiem truc tiep
  stats     Xem thong ke
""",
    )
    sub = p.add_subparsers(dest="command")

    c = sub.add_parser("chat", help="Chat voi agent")
    c.add_argument("--voice", action="store_true", help="Bat voice mode")
    c.add_argument("--shallow", action="store_true", help="Tra loi ngan")
    c.set_defaults(func=cmd_chat)

    w = sub.add_parser("web", help="Mo web UI")
    w.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    w.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    w.add_argument("--share", action="store_true", help="Share link")
    w.set_defaults(func=cmd_web)

    r = sub.add_parser("research", help="Nghien cuu sau")
    r.add_argument("topic", help="Chu de nghien cuu")
    r.add_argument("--depth", choices=["quick", "full", "deep"], default="full")
    r.add_argument("--export", action="store_true", help="Xuat thanh markdown")
    r.set_defaults(func=cmd_research)

    a = sub.add_parser("analyze", help="Phan tich ky thuat")
    a.add_argument("question", help="Cau hoi phan tich")
    a.set_defaults(func=cmd_analyze)

    g = sub.add_parser("graph", help="Knowledge graph")
    g.add_argument("--topic", help="Tim topic lien quan")
    g.add_argument("--export-md", action="store_true", help="Xuat markdown")
    g.set_defaults(func=cmd_graph)

    e = sub.add_parser("export", help="Xuat bao cao")
    e.add_argument("what", choices=["all", "research", "library", "graph"], default="all", nargs="?")
    e.set_defaults(func=cmd_export)

    i = sub.add_parser("ingest", help="Index docs vao knowledge base")
    i.add_argument("--category", default=None, help="Chi index category")
    i.add_argument("--url", default=None, help="Tai va index 1 URL")
    i.add_argument("--url-category", default=None, help="Category cho URL")
    i.add_argument("--file", default=None, help="Index 1 file local")
    i.set_defaults(func=cmd_ingest)

    s = sub.add_parser("search", help="Tim kiem trong knowledge base")
    s.add_argument("query")
    s.add_argument("-n", type=int, default=5)
    s.set_defaults(func=cmd_search)

    st = sub.add_parser("stats", help="Xem thong ke")
    st.set_defaults(func=cmd_stats)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
