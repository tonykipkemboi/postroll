"""
run.py — Main entry point for the postroll pipeline.

Usage:
  python run.py <youtube_url> [--force]

Examples:
  python run.py "https://www.youtube.com/watch?v=ibYkJVE1Rqs"
  python run.py "https://www.youtube.com/watch?v=ibYkJVE1Rqs" --force
"""
import sys
import re
import json
import argparse
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    Progress, TextColumn, BarColumn,
    MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich import box
from rich.live import Live
from rich.padding import Padding
from rich.markup import escape

from pipeline.ingest     import download
from pipeline.transcribe import transcribe
from pipeline.frames     import extract_frames
from pipeline.generate   import generate_section, _render_html
from config              import VIDEOS_DIR, OUTPUTS_DIR

console = Console()

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_ICONS   = {
    "pending": "[dim]○[/dim]",
    "done":    "[bold green]✓[/bold green]",
    "error":   "[bold red]✗[/bold red]",
}


def _icon(status: str) -> str:
    if status == "running":
        frame = _SPINNER[int(time.time() * 10) % len(_SPINNER)]
        return f"[bold yellow]{frame}[/bold yellow]"
    return _ICONS.get(status, "?")


# ── URL helpers ───────────────────────────────────────────────────────────────

def sanitize_youtube_url(url: str) -> str:
    url = url.strip().strip("'\"")
    url = re.sub(r"https?://youtu\.be/([A-Za-z0-9_-]{11})(.*)",
                 lambda m: f"https://www.youtube.com/watch?v={m.group(1)}", url)
    parsed = urlparse(url)
    if not re.search(r"(youtube\.com|youtu\.be)", parsed.netloc):
        raise ValueError(f"Not a YouTube URL: {url!r}")
    params  = parse_qs(parsed.query, keep_blank_values=False)
    video_id = params.get("v", [None])[0]
    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError(f"Could not extract a valid video ID from: {url!r}")
    return urlunparse(("https", "www.youtube.com", "/watch", "",
                       urlencode({"v": video_id}), ""))


def fmt_duration(seconds: float) -> str:
    h, m = divmod(int(seconds), 3600)
    m, s = divmod(m, 60)
    return f"{h}h {m}m" if h else f"{m}m {s}s"


# ── Renderables ───────────────────────────────────────────────────────────────

def _header(url: str) -> Panel:
    t = Text()
    t.append("  ◆ ", style="bold red")
    t.append("postroll", style="bold white")
    t.append("  companion guide generator\n", style="dim")
    t.append(f"    {url}", style="dim")
    return Panel(t, border_style="red", padding=(0, 1), box=box.ROUNDED)


def _pipeline_panel(steps: list, progress: Progress, show_progress: bool) -> Panel:
    tbl = Table(box=None, show_header=False, show_edge=False,
                padding=(0, 2), expand=False)
    tbl.add_column(width=3)
    tbl.add_column(min_width=13, no_wrap=True)
    tbl.add_column(min_width=52)
    tbl.add_column(width=7, justify="right")

    for step in steps:
        status = step["status"]
        if status == "pending":
            name    = Text(step["name"], style="dim")
            detail  = Text("")
            elapsed = Text("")
        else:
            name    = Text(step["name"],
                           style="bold white" if status == "running" else "bold")
            detail  = Text.from_markup(step.get("detail", ""))
            elapsed = Text(f"{step['elapsed']:.1f}s" if step.get("elapsed") else "",
                           style="dim")
        tbl.add_row(_icon(status), name, detail, elapsed)

    rows: list = [Padding(tbl, (1, 0))]
    if show_progress:
        rows.append(Padding(progress, (0, 2, 1, 6)))

    return Panel(Group(*rows), box=box.ROUNDED, border_style="dim", padding=(0, 1))


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(url: str, force: bool = False):
    url     = sanitize_youtube_url(url)
    t_total = time.time()

    console.print()
    console.print(_header(url))
    console.print()

    steps = [
        {"name": "INGEST",     "status": "pending", "detail": "", "elapsed": None},
        {"name": "TRANSCRIBE", "status": "pending", "detail": "", "elapsed": None},
        {"name": "FRAMES",     "status": "pending", "detail": "", "elapsed": None},
        {"name": "GENERATE",   "status": "pending", "detail": "", "elapsed": None},
    ]
    state = {"show_progress": False}

    progress = Progress(
        TextColumn("  {task.description}", style="dim"),
        BarColumn(bar_width=30, style="bright_black", complete_style="green"),
        MofNCompleteColumn(),
        TextColumn("[dim]·[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]·[/dim]"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        expand=False,
    )

    def render():
        return _pipeline_panel(steps, progress, state["show_progress"])

    with Live(render(), console=console, refresh_per_second=12) as live:

        # ── 1 · Ingest ─────────────────────────────────────────────────────────
        steps[0]["status"] = "running"
        live.update(render())
        t = time.time()

        video_dir = download(url, force=force)
        meta = json.loads((video_dir / "meta.json").read_text())
        dur  = fmt_duration(meta.get("duration_s", 0))

        steps[0].update(status="done", elapsed=time.time() - t,
                        detail=f"[white]{escape(meta['title'][:52])}[/white]"
                               f"  [dim]{dur}[/dim]")
        live.update(render())

        # ── 2 · Transcribe ─────────────────────────────────────────────────────
        steps[1]["status"] = "running"
        live.update(render())
        t = time.time()

        transcript = transcribe(video_dir, url, force=force)
        chapters   = transcript["chapters"]

        steps[1].update(status="done", elapsed=time.time() - t,
                        detail=f"[white]{len(chapters)} chapters[/white]"
                               f"  [dim]via {transcript['source']}[/dim]")
        live.update(render())

        # ── 3 · Extract frames ─────────────────────────────────────────────────
        steps[2]["status"] = "running"
        live.update(render())
        t = time.time()

        chapters = extract_frames(video_dir, chapters, force=force)

        steps[2].update(status="done", elapsed=time.time() - t,
                        detail=f"[white]{len(chapters)} frames[/white]"
                               f"  [dim]extracted[/dim]")
        live.update(render())

        # ── 4 · Generate guide ─────────────────────────────────────────────────
        steps[3]["status"] = "running"
        gen_task = progress.add_task("initializing…", total=len(chapters))
        state["show_progress"] = True
        live.update(render())
        t = time.time()

        out_path = _run_generate(
            video_dir, chapters, meta,
            progress=progress, task=gen_task,
            live=live, render=render,
            force=force,
        )

        steps[3].update(status="done", elapsed=time.time() - t,
                        detail=f"[white]{len(chapters)} sections[/white]"
                               f"  [dim]{out_path.name}[/dim]")
        state["show_progress"] = False
        live.update(render())

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    console.print()

    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2), expand=False)
    tbl.add_column(style="dim", min_width=10)
    tbl.add_column(style="white")
    tbl.add_row("Video",    meta["title"])
    tbl.add_row("Duration", dur)
    tbl.add_row("Chapters", str(len(chapters)))
    tbl.add_row("Output",   str(out_path))
    tbl.add_row("Time",     fmt_duration(elapsed))

    console.print(Panel(
        tbl,
        title="[bold green]✓  complete[/bold green]",
        border_style="green",
        box=box.ROUNDED,
        padding=(0, 1),
    ))
    console.print()


# ── Generation worker ─────────────────────────────────────────────────────────

def _run_generate(video_dir, chapters, meta, progress, task, live, render, force=False):
    """Generate the companion guide HTML with live progress updates."""
    output_dir = video_dir / "output"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "companion_guide.html"

    if out_path.exists() and not force:
        progress.update(task, description="cached — skipping",
                        completed=len(chapters))
        live.update(render())
        return out_path

    video_id = meta["video_id"]
    title    = meta["title"]
    url      = meta["url"]

    cache_dir = output_dir / "sections_cache"
    cache_dir.mkdir(exist_ok=True)

    sections = []
    for i, chapter in enumerate(chapters):
        cache_file = cache_dir / f"section_{i:04d}.json"

        progress.update(task, description=escape(chapter["title"][:58]))
        live.update(render())

        if cache_file.exists():
            section = json.loads(cache_file.read_text())
        else:
            section = generate_section(chapter, video_id, i, len(chapters))
            cache_file.write_text(json.dumps(section))

        sections.append(section)
        progress.advance(task)
        live.update(render())

    html = _render_html(sections, title, url, video_id)
    out_path.write_text(html)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    slug      = title.lower().replace(" ", "-")[:60]
    published = OUTPUTS_DIR / f"{slug}.html"
    published.write_text(html)

    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a companion guide for a YouTube video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n  python run.py "https://www.youtube.com/watch?v=zduSFxRajkE"'
    )
    parser.add_argument("url",     help="YouTube video URL")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if cached")
    args = parser.parse_args()

    try:
        run(args.url, force=args.force)
    except KeyboardInterrupt:
        console.print("\n[yellow]  Interrupted.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)
