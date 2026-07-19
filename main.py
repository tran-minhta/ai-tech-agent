"""AI Tech Knowledge Agent — hoi thoai ve lap trinh, OS, va technology voi long-term memory.

Cach dung:
    python main.py chat                    # CLI chat mode
    python main.py chat --voice            # Voice + CLI chat
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
        "Hoi bat cu dieu gi ve lap trinh, OS, hay cong nghe!\n"
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
    """Hien thi response tu brain."""
    text = result["text"]
    action = result.get("action")

    if action:
        _handle_action(action, speaker)

    md = Markdown(text)
    console.print(Panel(md, title="[bold cyan]AI Response[/]", border_style="cyan"))


def _handle_action(action: dict, speaker: Speaker | None = None):
    """Xu ly action tu LLM (save note, save snippet, status)."""
    act = action.get("action", "")

    if act == "save_note":
        # Note da duoc LLM generate, hien thi thong bao
        console.print("[green]Da luu note.[/]")
        if speaker:
            speaker.say("Da luu note.")

    elif act == "save_snippet":
        console.print("[green]Da luu snippet.[/]")
        if speaker:
            speaker.say("Da luu snippet.")

    elif act == "status":
        pass


def _show_stats(brain: Brain, speaker: Speaker | None = None):
    """Hien thi thong ke knowledge base."""
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
    """Tim kiem tu vector store."""
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
    """Luu note tu user."""
    if not text:
        console.print("[yellow]Nhap noi dung note.[/]")
        return
    note_id = struct_mem.add_note(title=text[:50], content=text, category="user")
    console.print(f"[green]Da luu note #{note_id}[/]")
    if speaker:
        speaker.say(f"Da luu note so {note_id}.")


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
    p = argparse.ArgumentParser(description="AI Tech Knowledge Agent")
    sub = p.add_subparsers(dest="command")

    c = sub.add_parser("chat", help="Chat voi agent")
    c.add_argument("--voice", action="store_true", help="Bat voice mode")
    c.add_argument("--shallow", action="store_true", help="Tra loi ngan (khong deep)")
    c.set_defaults(func=cmd_chat)

    i = sub.add_parser("ingest", help="Index docs vao knowledge base")
    i.add_argument("--category", default=None, help="Chi index category (python,linux,shell...)")
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
