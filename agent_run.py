#!/usr/bin/env python3
# run with: python3 agent_run.py <youtube-url> [--force]
"""
Entry point for the agent-based YouTube → HTML pipeline.

Usage:
  python agent_run.py <youtube-url> [--force]

Examples:
  python agent_run.py https://www.youtube.com/watch?v=J57nXAQozVA
  python agent_run.py 'https://youtu.be/J57nXAQozVA' --force
"""
import asyncio
import re
import sys
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


# ── URL sanitizer ─────────────────────────────────────────────────────────────

def sanitize_youtube_url(url: str) -> str:
    url = url.strip().strip("'\"")
    # Expand youtu.be short links
    url = re.sub(
        r"https?://youtu\.be/([A-Za-z0-9_-]{11})(.*)",
        lambda m: f"https://www.youtube.com/watch?v={m.group(1)}",
        url,
    )
    parsed = urlparse(url)
    if not re.search(r"(youtube\.com|youtu\.be)", parsed.netloc):
        raise ValueError(f"Not a YouTube URL: {url!r}")
    params   = parse_qs(parsed.query, keep_blank_values=False)
    video_id = params.get("v", [None])[0]
    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError(f"Could not extract a valid video ID from: {url!r}")
    clean_query = urlencode({"v": video_id})
    return urlunparse(("https", "www.youtube.com", "/watch", "", clean_query, ""))


# ── main ──────────────────────────────────────────────────────────────────────

async def main():
    args  = sys.argv[1:]
    force = "--force" in args
    urls  = [a for a in args if not a.startswith("--")]

    if not urls:
        print("Usage: python agent_run.py <youtube-url> [--force]")
        sys.exit(1)

    url = sanitize_youtube_url(urls[0])
    print(f"[agent_run] URL: {url}")

    from agent_pipeline.orchestrator import run_pipeline
    out = await run_pipeline(url, force=force)
    print(f"\n✓ Output: {out}")


if __name__ == "__main__":
    asyncio.run(main())
