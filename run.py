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
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich import box
from rich.rule import Rule
from rich.padding import Padding
import time

from pipeline.ingest     import download, get_video_id
from pipeline.transcribe import transcribe
from pipeline.frames     import extract_frames
from pipeline.generate   import generate_guide
from config              import VIDEOS_DIR

console = Console()


def sanitize_youtube_url(url: str) -> str:
    url = url.strip().strip("'\"")
    url = re.sub(r"https?://youtu\.be/([A-Za-z0-9_-]{11})(.*)",
                 lambda m: f"https://www.youtube.com/watch?v={m.group(1)}", url)
    parsed = urlparse(url)
    if not re.search(r"(youtube\.com|youtu\.be)", parsed.netloc):
        raise ValueError(f"Not a YouTube URL: {url!r}")
    params = parse_qs(parsed.query, keep_blank_values=False)
    video_id = params.get("v", [None])[0]
    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError(f"Could not extract a valid video ID from: {url!r}")
    clean_query = urlencode({"v": video_id})
    clean = urlunparse(("https", "www.youtube.com", "/watch", "", clean_query, ""))
    return clean


def fmt_duration(seconds: float) -> str:
    h, m = divmod(int(seconds), 3600)
    m, s = divmod(m, 60)
    if h:
        return f"{h}h {m}m"
    return f"{m}m {s}s"


def run(url: str, force: bool = False):
    url = sanitize_youtube_url(url)
    start_time = time.time()

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        f"[bold white]postroll[/bold white]  [dim]companion guide generator[/dim]\n"
        f"[dim]{url}[/dim]",
        border_style="bright_red",
        padding=(0, 2),
    ))
    console.print()

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    console.print(Rule("[bold]Step 1 — Ingest[/bold]", style="dim"))
    console.print()

    with console.status("[dim]Fetching metadata and downloading...[/dim]", spinner="dots"):
        video_dir = download(url, force=force)
        meta = json.loads((video_dir / "meta.json").read_text())

    duration_str = fmt_duration(meta.get("duration_s", 0))
    console.print(f"  [green]✓[/green] [bold]{meta['title']}[/bold]  [dim]{duration_str}[/dim]")
    console.print()

    # ── Step 2: Transcribe ────────────────────────────────────────────────────
    console.print(Rule("[bold]Step 2 — Transcribe[/bold]", style="dim"))
    console.print()

    with console.status("[dim]Transcribing via Gemini...[/dim]", spinner="dots"):
        transcript = transcribe(video_dir, url, force=force)

    chapters = transcript["chapters"]
    source   = transcript["source"]
    console.print(f"  [green]✓[/green] [bold]{len(chapters)} chapters[/bold]  [dim]via {source}[/dim]")
    console.print()

    # ── Step 3: Extract frames ────────────────────────────────────────────────
    console.print(Rule("[bold]Step 3 — Extract frames[/bold]", style="dim"))
    console.print()

    with console.status("[dim]Extracting keyframes with ffmpeg...[/dim]", spinner="dots"):
        chapters = extract_frames(video_dir, chapters, force=force)

    console.print(f"  [green]✓[/green] [bold]{len(chapters)} frames[/bold] extracted")
    console.print()

    # ── Step 4: Generate guide ────────────────────────────────────────────────
    console.print(Rule("[bold]Step 4 — Generate guide[/bold]", style="dim"))
    console.print()

    # Patch generate_guide to report progress via Rich
    out_path = _generate_with_progress(video_dir, chapters, meta, force=force)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    console.print()

    summary = Table(box=box.ROUNDED, show_header=False, border_style="dim", padding=(0, 1))
    summary.add_column(style="dim")
    summary.add_column(style="white")
    summary.add_row("Video",    meta["title"])
    summary.add_row("Duration", duration_str)
    summary.add_row("Chapters", str(len(chapters)))
    summary.add_row("Output",   str(out_path))
    summary.add_row("Time",     fmt_duration(elapsed))

    console.print(Panel(
        summary,
        title="[bold green]✓ Done[/bold green]",
        border_style="green",
        padding=(0, 1),
    ))
    console.print()


def _generate_with_progress(video_dir, chapters, meta, force=False):
    """Wrap generate_guide with a Rich progress bar."""
    from pipeline.generate import generate_section, _render_html
    from config import OUTPUTS_DIR
    import json as _json

    output_dir = video_dir / "output"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "companion_guide.html"

    if out_path.exists() and not force:
        console.print("  [dim]Guide already exists, skipping generation.[/dim]")
        return out_path

    video_id = meta["video_id"]
    title    = meta["title"]
    url      = meta["url"]

    cache_dir = output_dir / "sections_cache"
    cache_dir.mkdir(exist_ok=True)

    sections = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=32),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"  Generating {len(chapters)} sections", total=len(chapters))

        for i, chapter in enumerate(chapters):
            cache_file = cache_dir / f"section_{i:04d}.json"
            progress.update(task, description=f"  [dim]{chapter['title'][:52]}[/dim]")

            if cache_file.exists():
                section = _json.loads(cache_file.read_text())
            else:
                section = generate_section(chapter, video_id, i, len(chapters))
                cache_file.write_text(_json.dumps(section))

            sections.append(section)
            progress.advance(task)

    html = _render_html(sections, title, url, video_id)
    out_path.write_text(html)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    slug = title.lower().replace(" ", "-")[:60]
    published = OUTPUTS_DIR / f"{slug}.html"
    published.write_text(html)

    console.print(f"  [green]✓[/green] Saved → [cyan]{published}[/cyan]")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a companion guide for a YouTube video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n  python run.py "https://www.youtube.com/watch?v=zduSFxRajkE"'
    )
    parser.add_argument("url",     help="YouTube video URL")
    parser.add_argument("--force", action="store_true", help="Re-run all steps even if cached")
    args = parser.parse_args()

    try:
        run(args.url, force=args.force)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)
