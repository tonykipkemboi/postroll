"""
Orchestrator + section subagents.

Flow:
  1. ingest_video       (sequential)
  2. transcribe_video   (sequential, Gemini 2.5 Pro)
  3. run_section_agent  (PARALLEL, up to SECTION_CONCURRENCY at once)
     Each subagent:
       a. extract_frame    (ffmpeg)
       b. generate_prose   (Gemini Flash by default; swap to Claude by setting
                            USE_CLAUDE_FOR_SECTIONS=True in config.py + API key)
  4. assemble_html      (sequential)

When USE_CLAUDE_FOR_SECTIONS is True and ANTHROPIC_API_KEY is set, each
section agent becomes a Claude subagent (anthropic.AsyncAnthropic).
"""
import asyncio
import json
import traceback
from pathlib import Path

from agent_pipeline.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_SECTION_MODEL,
    GEMINI_API_KEY,
    GEMINI_WRITE_MODEL,
    OUTPUTS_DIR,
    SECTION_CONCURRENCY,
    USE_CLAUDE_FOR_SECTIONS,
)
from agent_pipeline.tools.ingest     import ingest_video
from agent_pipeline.tools.transcribe import transcribe_video
from agent_pipeline.tools.frame      import extract_frame
from agent_pipeline.tools.assemble   import assemble_html


# ── prompt ────────────────────────────────────────────────────────────────────

SECTION_SYSTEM = """You are writing one section of a deep technical companion guide for a YouTube video.
A video clip covering exactly this section is attached — watch it carefully before writing.

Return a JSON object with exactly two keys:
  "prose_html"  – standalone HTML (no <html>/<head>/<body>). Use <p>, <pre><code>, <ul>, <strong>, <em>.
  "use_frame"   – boolean: true only if the screenshot shows code, terminal output, or a non-trivial diagram.

Strict rules for prose_html:
- Write 3-5 substantive paragraphs. Prioritise depth over breadth.
- Be SPECIFIC: include exact numbers, variable names, function names, and vocabulary the speaker uses.
  Bad: "the model has a large vocabulary"
  Good: "GPT-4 uses the cl100k_base tokenizer with a vocabulary of 100,277 tokens"
- Cite papers and tools by name when the speaker references them (e.g. "GPT-2 paper Section 2.2", "tiktoken", "SentencePiece").
- If the video shows code on screen, transcribe the relevant snippet EXACTLY inside
  <pre><code class="language-python"> ... </code></pre>. Do not paraphrase code.
- Give concrete examples with actual values from the video (e.g. "the string '127' is one token, but '677' splits into ' 6' and '77'").
- Do NOT define tokenization at the start of every section — assume the reader has read the previous sections.
- Do NOT include an <h2> heading or any YouTube links (both added by the template).
- BANNED words and phrases: "delves", "delve into", "explores", "unpacks", "examines", "in this section",
  "it's worth noting", "crucial", "it is important to note", "comprehensive", "nuanced", "fascinating",
  "highlights the importance", "plays a key role", "significant", "notably", "importantly".
  Use plain, direct language instead.
- No markdown — pure HTML only.

Return raw JSON only. No markdown fences."""


def _section_user_text(chapter: dict) -> str:
    return (
        f'Chapter: "{chapter["title"]}"\n'
        f'Timestamp: {chapter["start"]:.1f}s – {chapter["end"]:.1f}s  '
        f'({chapter["end"] - chapter["start"]:.0f}s)\n'
        f'Summary hint: {chapter.get("summary", "")}\n'
        f'has_code: {chapter.get("has_code", False)}\n'
        f'has_diagram: {chapter.get("has_diagram", False)}\n\n'
        f'Watch the attached video clip and write the companion guide section for this chapter. '
        f'Be specific — extract exact numbers, code snippets, paper references, and examples '
        f'directly from what the speaker says and shows on screen.'
    )


def _parse_section_response(raw: str, frame_b64: str | None) -> tuple[str, bool]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        data      = json.loads(raw)
        prose     = data.get("prose_html", raw)
        use_frame = bool(data.get("use_frame", False))
    except json.JSONDecodeError as e:
        print(f"  [JSON parse failed: {e}; using raw text as prose]")
        prose     = raw
        use_frame = frame_b64 is not None
    return prose, use_frame


# ── video clip helpers ────────────────────────────────────────────────────────

async def _extract_clip(video_dir: Path, start: float, end: float) -> Path | None:
    """Cut a video segment with ffmpeg. Returns path or None on failure."""
    clip_path = video_dir / "clips" / f"clip_{int(start)}_{int(end)}.mp4"
    if clip_path.exists():
        return clip_path
    clip_path.parent.mkdir(parents=True, exist_ok=True)

    video_path = video_dir / "video.mp4"
    if not video_path.exists():
        return None

    duration = end - start
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264", "-crf", "28", "-preset", "fast",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        str(clip_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.communicate()
    return clip_path if clip_path.exists() else None


def _upload_clip_sync(clip_path: Path) -> str | None:
    """Upload a video clip to Gemini Files API. Returns URI or None."""
    import google.genai as genai
    import time

    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        uploaded = client.files.upload(file=str(clip_path), config={"mime_type": "video/mp4"})
        for _ in range(30):
            if uploaded.state.name == "ACTIVE":
                return uploaded.uri
            time.sleep(3)
            uploaded = client.files.get(name=uploaded.name)
        # timed out
        client.files.delete(name=uploaded.name)
        return None
    except Exception as e:
        print(f"  [clip upload failed: {e}]")
        return None


def _delete_clip_sync(uri: str) -> None:
    """Best-effort cleanup of uploaded Gemini file."""
    import google.genai as genai
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        # extract file name from URI (files/xxxx)
        name = "/".join(uri.split("/")[-2:])
        client.files.delete(name=name)
    except Exception:
        pass


# ── writer backends ───────────────────────────────────────────────────────────

async def _write_with_gemini(
    chapter: dict,
    frame_b64: str | None,
    video_dir: Path,
) -> tuple[str, bool]:
    """Section prose via Gemini Pro. Uploads a video clip for full visual context."""
    import google.genai as genai
    from google.genai import types

    # Extract and upload the video clip for this section
    clip_uri: str | None = None
    try:
        clip_path = await _extract_clip(video_dir, chapter["start"], chapter["end"])
        if clip_path:
            loop = asyncio.get_running_loop()
            clip_uri = await loop.run_in_executor(None, _upload_clip_sync, clip_path)
    except Exception as e:
        print(f"  [clip prep failed: {e}]")

    def _sync():
        client = genai.Client(api_key=GEMINI_API_KEY)
        parts: list = []

        if clip_uri:
            # Full video clip — best quality
            parts.append(types.Part(file_data=types.FileData(file_uri=clip_uri, mime_type="video/mp4")))
        elif frame_b64:
            # Fallback: single JPEG frame
            import base64
            parts.append(types.Part.from_bytes(data=base64.b64decode(frame_b64), mime_type="image/jpeg"))

        parts.append(types.Part(text=_section_user_text(chapter)))
        response = client.models.generate_content(
            model=GEMINI_WRITE_MODEL,
            contents=types.Content(parts=parts),
            config=types.GenerateContentConfig(
                system_instruction=SECTION_SYSTEM,
                temperature=0.3,
            ),
        )
        return response.text

    loop = asyncio.get_running_loop()
    try:
        raw = await loop.run_in_executor(None, _sync)
    finally:
        # Always clean up the uploaded clip, even if prose generation fails
        if clip_uri:
            await loop.run_in_executor(None, _delete_clip_sync, clip_uri)

    return _parse_section_response(raw, frame_b64)


async def _write_with_claude(chapter: dict, frame_b64: str | None, video_dir: Path) -> tuple[str, bool]:
    """Section prose via Claude (async). Requires ANTHROPIC_API_KEY in env."""
    import anthropic

    content: list[dict] = []
    if frame_b64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_b64},
        })
    content.append({"type": "text", "text": _section_user_text(chapter)})

    client   = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model=ANTHROPIC_SECTION_MODEL,
        max_tokens=2048,
        system=SECTION_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    return _parse_section_response(response.content[0].text, frame_b64)


async def _write_prose(chapter: dict, frame_b64: str | None, video_dir: Path) -> tuple[str, bool]:
    """Route to Claude or Gemini based on config."""
    if USE_CLAUDE_FOR_SECTIONS and ANTHROPIC_API_KEY:
        return await _write_with_claude(chapter, frame_b64, video_dir)
    return await _write_with_gemini(chapter, frame_b64, video_dir)


# ── section subagent ──────────────────────────────────────────────────────────

_sem = asyncio.Semaphore(SECTION_CONCURRENCY)


async def run_section_agent(
    idx: int,
    chapter: dict,
    video_dir: Path,
    video_id: str,
    force: bool = False,
) -> dict:
    """
    One section subagent — runs in parallel with all other sections.
    Writes/reads cache at video_dir/sections/{idx:02d}.json for resume support.
    """
    cache_path = video_dir / "sections" / f"{idx:02d}.json"

    if not force and cache_path.exists():
        print(f"  [{idx:02d}] cached ✓")
        return json.loads(cache_path.read_text())

    async with _sem:
        title    = chapter["title"]
        midpoint = (chapter["start"] + chapter["end"]) / 2.0

        print(f"  [{idx:02d}] frame   → {title[:45]}")
        frame_b64 = await extract_frame(video_dir, midpoint)

        writer = "Claude" if (USE_CLAUDE_FOR_SECTIONS and ANTHROPIC_API_KEY) else "Gemini"
        print(f"  [{idx:02d}] {writer:6s}  → {title[:45]}")
        try:
            prose_html, use_frame = await _write_prose(chapter, frame_b64, video_dir)
        except Exception as e:
            traceback.print_exc()
            # Don't expose internal details (paths, keys) in user-facing HTML
            prose_html = "<p><em>[Section generation failed — see console for details.]</em></p>"
            use_frame  = False

        section = {
            "index":      idx,
            "chapter":    chapter,
            "prose_html": prose_html,
            "frame_b64":  frame_b64,   # always keep; assemble decides display
            "use_frame":  use_frame,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to temp then rename, so a killed process never leaves corrupt JSON
        import tempfile, os as _os
        tmp_fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".tmp")
        try:
            with _os.fdopen(tmp_fd, "w") as f:
                f.write(json.dumps(section))
            _os.replace(tmp_path, cache_path)
        except Exception:
            _os.unlink(tmp_path)
            raise
        print(f"  [{idx:02d}] done ✓")
        return section


# ── main pipeline ─────────────────────────────────────────────────────────────

async def run_pipeline(url: str, force: bool = False) -> Path:
    """
    Orchestrate the full pipeline:
      ingest → transcribe → parallel section agents → assemble HTML.
    Returns path to the generated HTML file.
    """
    # 1. Ingest (sequential)
    video_dir, meta = await ingest_video(url, force=force)

    # 2. Transcribe (sequential — must come after ingest)
    chapters = await transcribe_video(
        video_dir, url, meta["duration"], force=force
    )

    # 3. Section agents — ALL spawned at once, throttled by semaphore
    writer = "Claude" if (USE_CLAUDE_FOR_SECTIONS and ANTHROPIC_API_KEY) else "Gemini Flash"
    print(f"\n[pipeline] spawning {len(chapters)} section agents "
          f"(writer={writer}, concurrency={SECTION_CONCURRENCY})")

    tasks   = [
        run_section_agent(i, ch, video_dir, meta["id"], force=force)
        for i, ch in enumerate(chapters)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Degrade gracefully on individual failures
    sections = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"  [{i:02d}] ERROR: {r}")
            sections.append({
                "index": i,
                "chapter": chapters[i],
                "prose_html": f"<p><em>[Error in section {i}: {r}]</em></p>",
                "frame_b64": None,
                "use_frame": False,
            })
        else:
            sections.append(r)

    sections.sort(key=lambda s: s["index"])

    # 4. Assemble (sequential)
    print(f"\n[pipeline] assembling HTML ({len(sections)} sections)...")
    html = assemble_html(sections, meta)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"{meta['slug']}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[pipeline] ✓  {out_path}")
    return out_path
