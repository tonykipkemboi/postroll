"""
Tool: extract_frame
Uses ffmpeg to grab a single frame from the video at the given timestamp.
Returns base64-encoded JPEG string, or None on failure.
"""
import asyncio
import base64
import tempfile
from pathlib import Path


async def extract_frame(video_dir: Path, timestamp: float) -> str | None:
    """
    Extract a frame at *timestamp* seconds from the video in *video_dir*.
    Returns base64 JPEG string, or None if extraction fails.
    """
    video_files = list(video_dir.glob("video.*"))
    if not video_files:
        return None

    video_path = video_files[0]

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        out_path = tmp.name

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "3",
        "-vf", "scale=1280:-1",
        out_path,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,  # capture for diagnostics
    )
    _, stderr_bytes = await proc.communicate()

    out = Path(out_path)
    try:
        if proc.returncode != 0:
            if stderr_bytes:
                print(f"  [ffmpeg frame failed (rc={proc.returncode}): {stderr_bytes.decode(errors='replace').strip()[-200:]}]")
            return None
        if not out.exists() or out.stat().st_size == 0:
            return None
        return base64.b64encode(out.read_bytes()).decode()
    finally:
        out.unlink(missing_ok=True)  # always clean up temp file
