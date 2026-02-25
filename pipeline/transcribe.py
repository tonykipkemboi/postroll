"""
transcribe.py — Transcription via Gemini (primary) or Whisper (fallback).

Gemini path:   tries YouTube URL first; if coverage < 80% of known duration,
               uploads the local video file via Files API for full coverage.
               For videos > CHUNK_THRESHOLD seconds, splits audio into chunks
               to stay within Gemini's 1M token context limit.
Whisper path:  processes local audio.mp3 — used if no Gemini key.

Output: videos/{video_id}/transcript.json
"""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from config import VIDEOS_DIR, GEMINI_API_KEY, GEMINI_INGEST_MODEL, WHISPER_MODEL

# Videos longer than this (seconds) are split into chunks before upload
CHUNK_THRESHOLD = 3600       # 60 minutes
CHUNK_SIZE      = 1500       # 25 minutes per chunk


TRANSCRIPT_PROMPT = """Analyse this video and return a JSON object with this exact schema:
{
  "source": "gemini",
  "language": "<detected language code>",
  "chapters": [
    {
      "title": "<concise chapter title>",
      "start": <start seconds as float>,
      "end": <end seconds as float>,
      "transcript": "<verbatim transcript of this segment>",
      "speaker": "<speaker name or role if identifiable, else null>",
      "has_screen_content": <true if code/diagram/slide is visible, false if talking head only>,
      "screen_content_type": "<'code' | 'diagram' | 'terminal' | 'slide' | 'none'>"
    }
  ],
  "full_text": "<full verbatim transcript of the entire video>"
}

Chapter detection rules:
- IMPORTANT: cover the ENTIRE video from start to finish, do not stop early
- Split on natural topic changes, not fixed time intervals
- Minimum chapter length: 30 seconds
- Maximum chapter length: 3 minutes
- A chapter should cover one coherent concept or step
- The last chapter's end time must match the video's total duration

Return ONLY the JSON, no markdown fences."""


def transcribe(video_dir: Path, url: str, force: bool = False) -> dict:
    out_path = video_dir / "transcript.json"

    if out_path.exists() and not force:
        print("Transcript already exists, loading.")
        return json.loads(out_path.read_text())

    meta_path = video_dir / "meta.json"
    known_duration = json.loads(meta_path.read_text()).get("duration_s", 0) if meta_path.exists() else 0

    if GEMINI_API_KEY:
        result = _transcribe_gemini(video_dir, url, known_duration)
    else:
        print("No Gemini key found — falling back to Whisper.")
        result = _transcribe_whisper(video_dir)

    # Clamp/drop chapters with timestamps beyond the actual video duration
    if known_duration > 0:
        result = _sanitize_timestamps(result, known_duration)

    out_path.write_text(json.dumps(result, indent=2))
    print(f"Transcript saved: {len(result['chapters'])} chapters")
    return result


def _sanitize_timestamps(result: dict, duration: float) -> dict:
    """
    Drop chapters that start past the video end.
    Clamp chapter end times to the video duration.
    """
    cleaned = []
    for c in result["chapters"]:
        if c["start"] >= duration:
            continue  # starts past video end — hallucinated
        c = dict(c)
        c["end"] = min(c["end"], duration)
        if c["end"] <= c["start"]:
            continue  # zero/negative duration after clamping — skip
        cleaned.append(c)
    if cleaned:
        result = dict(result)
        result["chapters"] = cleaned
    return result


# ── Gemini path ───────────────────────────────────────────────────────────────

def _transcribe_gemini(video_dir: Path, url: str, known_duration: float = 0) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    # ── Try YouTube URL first (fast, no upload needed) ────────────────────────
    print(f"Transcribing via Gemini ({GEMINI_INGEST_MODEL}) — YouTube URL...")
    try:
        result = _call_gemini(client, types.Part(
            file_data=types.FileData(file_uri=url)
        ), extra_prompt=TRANSCRIPT_PROMPT)

        last_end = result["chapters"][-1]["end"] if result["chapters"] else 0
        coverage = last_end / known_duration if known_duration else 1.0
        print(f"  YouTube URL coverage: {last_end:.0f}s / {known_duration}s ({coverage:.0%})")

        if coverage >= 0.80:
            return result
        else:
            print(f"  Coverage too low ({coverage:.0%}) — uploading local video file...")
    except Exception as e:
        print(f"  YouTube URL failed ({e}) — uploading local video file...")

    # ── Fall back: upload local audio file (chunked if long) ─────────────────
    audio_path = video_dir / "raw" / "audio.mp3"
    if not audio_path.exists():
        raise FileNotFoundError(f"No audio file found at {audio_path}")

    if known_duration > CHUNK_THRESHOLD:
        print(f"  Video is {known_duration/60:.0f} min — splitting into {CHUNK_SIZE//60}-min chunks...")
        return _transcribe_chunked(client, audio_path, known_duration)

    print(f"  Uploading {audio_path.name} ({audio_path.stat().st_size // 1_000_000} MB)...")
    return _upload_and_transcribe(client, audio_path, offset=0)


def _transcribe_chunked(client, audio_path: Path, known_duration: float) -> dict:
    """Split audio into chunks, transcribe each, then merge."""
    from google.genai import types

    chunks = []
    start = 0.0
    while start < known_duration:
        end = min(start + CHUNK_SIZE, known_duration)
        chunks.append((start, end))
        start = end

    print(f"  {len(chunks)} chunks: {[f'{s/60:.0f}-{e/60:.0f}m' for s, e in chunks]}")

    all_chapters = []
    all_text_parts = []

    for i, (start, end) in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}: {start/60:.0f}–{end/60:.0f} min")

        # Extract chunk via ffmpeg into a temp file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(tmp_fd)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(start), "-t", str(end - start),
                 "-i", str(audio_path), "-c", "copy", tmp_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            result = _upload_and_transcribe(client, Path(tmp_path), offset=start)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        all_chapters.extend(result.get("chapters", []))
        if result.get("full_text"):
            all_text_parts.append(result["full_text"])

    return {
        "source": "gemini",
        "language": "en",
        "chapters": all_chapters,
        "full_text": " ".join(all_text_parts),
    }


def _upload_and_transcribe(client, audio_path: Path, offset: float = 0) -> dict:
    """Upload a single audio file to Gemini Files API and transcribe it."""
    from google.genai import types

    print(f"    Uploading {audio_path.name} ({audio_path.stat().st_size // 1_000_000} MB)...")
    uploaded = client.files.upload(file=str(audio_path))

    upload_active = False
    for _ in range(60):
        if uploaded.state.name == "ACTIVE":
            upload_active = True
            break
        time.sleep(5)
        uploaded = client.files.get(name=uploaded.name)

    if not upload_active:
        raise TimeoutError("Gemini file upload never became ACTIVE after 300s")

    print(f"    File ready — transcribing (offset {offset:.0f}s)...")
    try:
        prompt = TRANSCRIPT_PROMPT
        if offset > 0:
            prompt = f"NOTE: This audio starts at {offset:.1f}s into the full video. All timestamps in your response must be offset by {offset:.1f}s.\n\n{TRANSCRIPT_PROMPT}"

        result = _call_gemini(client, types.Part(
            file_data=types.FileData(file_uri=uploaded.uri, mime_type="audio/mpeg")
        ), extra_prompt=prompt)
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

    return result


def _call_gemini(client, video_part, extra_prompt: str = None) -> dict:
    from google.genai import types

    prompt = extra_prompt if extra_prompt else TRANSCRIPT_PROMPT

    response = client.models.generate_content(
        model=GEMINI_INGEST_MODEL,
        contents=types.Content(parts=[
            video_part,
            types.Part(text=prompt),
        ]),
        config=types.GenerateContentConfig(
            temperature=0.1,
        ),
    )

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    return json.loads(raw)


# ── Whisper fallback ──────────────────────────────────────────────────────────

def _transcribe_whisper(video_dir: Path) -> dict:
    import whisper

    audio_path = video_dir / "raw" / "audio.mp3"
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    print(f"Transcribing via Whisper ({WHISPER_MODEL})...")
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(str(audio_path), word_timestamps=True, verbose=False)

    chapters = []
    current, start = [], result["segments"][0]["start"]
    for seg in result["segments"]:
        current.append(seg)
        if seg["end"] - start >= 90:
            chapters.append({
                "title": f"Section {len(chapters)+1}",
                "start": start,
                "end": seg["end"],
                "transcript": " ".join(s["text"] for s in current).strip(),
                "speaker": None,
                "has_screen_content": None,
                "screen_content_type": None,
            })
            current, start = [], seg["end"]
    if current:
        chapters.append({
            "title": f"Section {len(chapters)+1}",
            "start": start,
            "end": current[-1]["end"],
            "transcript": " ".join(s["text"] for s in current).strip(),
            "speaker": None,
            "has_screen_content": None,
            "screen_content_type": None,
        })

    return {
        "source": "whisper",
        "language": result.get("language", "en"),
        "chapters": chapters,
        "full_text": result["text"],
    }
