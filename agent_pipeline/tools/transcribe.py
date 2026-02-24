"""
Tool: transcribe_video
Uses Gemini 2.5 Pro to extract structured chapters from a YouTube video.
Falls back to file upload if YouTube URL coverage is < COVERAGE_THRESHOLD.
"""
import asyncio
import json
import time
from pathlib import Path

import google.genai as genai
from google.genai import types

from agent_pipeline.config import GEMINI_API_KEY, GEMINI_INGEST_MODEL, COVERAGE_THRESHOLD

# Videos longer than this use audio-only upload (video tokens = 258/s, audio = 32/s)
# 1M token limit / 258 ≈ 3876s ≈ 64 min — use 45 min as safe threshold
AUDIO_ONLY_THRESHOLD_S = 45 * 60

# ── prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(duration: float) -> str:
    # ~1 chapter per 5 minutes, clamped to reasonable range
    target = max(8, min(60, int(duration / 300)))
    return f"""
Analyze this YouTube video completely from start to finish. The video is {duration:.0f} seconds ({duration/60:.1f} minutes) long.

Return ONLY a JSON object with this exact structure:
{{
  "title": "...",
  "summary": "2-3 sentence overview",
  "chapters": [
    {{
      "index": 0,
      "title": "...",
      "start": 0.0,
      "end": 45.0,
      "summary": "...",
      "has_code": false,
      "has_diagram": false
    }}
  ]
}}

Rules:
- Cover the ENTIRE video from second 0 to second {duration:.0f}. Do NOT stop early.
- The last chapter's end time MUST equal {duration:.0f} exactly.
- Aim for ~{target} chapters of roughly equal length (~{duration/target:.0f}s each).
- has_code = true if the speaker mentions writing or running code in this chapter.
- has_diagram = true if the speaker describes a diagram, architecture, or visual concept.
- No markdown fences. Return raw JSON only.
""".strip()


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


def _call_gemini_sync(client: genai.Client, video_part, prompt: str) -> dict:
    response = client.models.generate_content(
        model=GEMINI_INGEST_MODEL,
        contents=types.Content(parts=[
            video_part,
            types.Part(text=prompt),
        ]),
        config=types.GenerateContentConfig(temperature=0.1),
    )
    raw = response.text.strip()
    # Strip markdown fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _coverage(result: dict, duration: float) -> float:
    if not duration or not result.get("chapters"):
        return 0.0
    last_end = max(c["end"] for c in result["chapters"])
    return last_end / duration


def _sanitize_timestamps(result: dict, duration: float) -> dict:
    cleaned = []
    for c in result["chapters"]:
        if c["start"] >= duration:
            continue  # hallucinated
        c = dict(c)
        c["end"] = min(c["end"], duration)
        if c["end"] <= c["start"]:
            continue
        cleaned.append(c)
    if cleaned:
        result = dict(result)
        result["chapters"] = cleaned
    return result


def _transcribe_sync(video_dir: Path, url: str, duration: float) -> dict:
    client = _make_client()
    prompt = _build_prompt(duration)

    # ── attempt 1: YouTube URL (fast) ────────────────────────────────────────
    print("[transcribe] trying YouTube URL ingestion...")
    video_part = types.Part(
        file_data=types.FileData(file_uri=url, mime_type="video/mp4")
    )
    try:
        result = _call_gemini_sync(client, video_part, prompt)
        cov = _coverage(result, duration)
        print(f"[transcribe] URL coverage: {cov:.0%}")
        if cov >= COVERAGE_THRESHOLD:
            return _sanitize_timestamps(result, duration)
    except Exception as e:
        print(f"[transcribe] URL attempt failed: {e}")

    # ── attempt 2: file upload ───────────────────────────────────────────────
    # For long videos (>45 min) use audio-only: 32 tokens/s vs 258 tokens/s for video.
    # audio.mp3 is always downloaded during ingest.
    if duration > AUDIO_ONLY_THRESHOLD_S:
        upload_candidates = list(video_dir.glob("audio.mp3"))
        mode = "audio"
    else:
        upload_candidates = list(video_dir.glob("video.*"))
        mode = "video"

    if not upload_candidates:
        # fallback: try the other type
        upload_candidates = list(video_dir.glob("video.*")) or list(video_dir.glob("audio.*"))
        mode = "fallback"

    if not upload_candidates:
        raise FileNotFoundError(f"No media file found in {video_dir}")

    upload_path = upload_candidates[0]
    print(f"[transcribe] uploading {upload_path.name} ({upload_path.stat().st_size / 1e6:.0f} MB, mode={mode})...")

    uploaded = client.files.upload(file=str(upload_path))

    # Wait for ACTIVE state
    upload_active = False
    for _ in range(60):
        if uploaded.state.name == "ACTIVE":
            upload_active = True
            break
        time.sleep(5)
        uploaded = client.files.get(name=uploaded.name)
    if not upload_active:
        raise TimeoutError("Gemini file upload never became ACTIVE after 300s")

    print("[transcribe] file active, generating chapters...")
    video_part = types.Part(file_data=types.FileData(file_uri=uploaded.uri))
    try:
        result = _call_gemini_sync(client, video_part, prompt)
    finally:
        client.files.delete(name=uploaded.name)

    cov = _coverage(result, duration)
    print(f"[transcribe] upload coverage: {cov:.0%}")
    if cov < COVERAGE_THRESHOLD:
        print(f"[transcribe] WARNING: coverage {cov:.0%} below threshold — output may be incomplete")

    return _sanitize_timestamps(result, duration)


# ── main tool ─────────────────────────────────────────────────────────────────

async def transcribe_video(
    video_dir: Path,
    url: str,
    duration: float,
    force: bool = False,
) -> list[dict]:
    """
    Return list of chapter dicts for the video.
    Caches result in video_dir/transcript.json.
    """
    cache_path = video_dir / "transcript.json"

    if not force and cache_path.exists():
        data = json.loads(cache_path.read_text())
        print(f"[transcribe] cached: {len(data['chapters'])} chapters")
        return data["chapters"]

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _transcribe_sync, video_dir, url, duration)

    cache_path.write_text(json.dumps(result, indent=2))
    n = len(result["chapters"])
    print(f"[transcribe] done → {n} chapters")
    return result["chapters"]
