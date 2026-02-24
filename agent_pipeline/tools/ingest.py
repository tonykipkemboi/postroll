"""
Tool: ingest_video
Downloads metadata, audio, and video for a YouTube URL via yt_dlp.
Returns (video_dir, meta_dict).
"""
import asyncio
import json
import re
from pathlib import Path

import yt_dlp as _yt

from agent_pipeline.config import VIDEOS_DIR


# ── helpers ───────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text[:80]


def _run_ydl(opts: dict, url: str):
    with _yt.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=True)


def _run_ydl_nodown(opts: dict, url: str):
    opts = {**opts, "quiet": True}
    with _yt.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


# ── main tool ─────────────────────────────────────────────────────────────────

async def ingest_video(url: str, force: bool = False) -> tuple[Path, dict]:
    """
    Download video metadata + files (audio, video) for *url*.
    Returns (video_dir, meta) where meta contains id, title, slug, duration, etc.
    Skips download if meta.json already exists (unless force=True).
    """
    # Fetch lightweight metadata first to get video ID / slug
    loop = asyncio.get_running_loop()
    info = await loop.run_in_executor(
        None, _run_ydl_nodown, {"quiet": True}, url
    )
    video_id   = info["id"]
    title      = info["title"]
    slug       = _slugify(title)
    duration   = float(info.get("duration") or 0)
    video_dir  = VIDEOS_DIR / slug

    meta_path  = video_dir / "meta.json"

    if not force and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"[ingest] cached: {slug}")
        return video_dir, meta

    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ingest] downloading: {title} ({duration:.0f}s)")

    # Download audio (for potential future use)
    audio_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(video_dir / "audio.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
    }
    await loop.run_in_executor(None, _run_ydl, audio_opts, url)

    # Download video (480p for Gemini file upload fallback + ffmpeg frames)
    video_opts = {
        "format": "bestvideo[height<=480]+bestaudio/best[height<=480]/best",
        "outtmpl": str(video_dir / "video.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
    }
    await loop.run_in_executor(None, _run_ydl, video_opts, url)

    # Find video file
    video_files = list(video_dir.glob("video.*"))
    if not video_files:
        raise FileNotFoundError(
            f"Video download failed: no video.* found in {video_dir}. "
            "Check yt_dlp output above for errors."
        )
    video_file = str(video_files[0])

    meta = {
        "id":       video_id,
        "title":    title,
        "slug":     slug,
        "duration": duration,
        "url":      url,
        "video_file": video_file,
        "description": info.get("description", "")[:500],
        "uploader":    info.get("uploader", ""),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[ingest] done → {video_dir}")
    return video_dir, meta
