"""
generate.py — Generate companion guide sections using an LLM.

For each chapter the model decides:
  - SCREENSHOT: chapter has meaningful screen content (code/diagram/terminal)
    → include the frame as a static image with a timestamp hyperlink
  - IFRAME: talking head or low-value visual
    → embed an inline YouTube iframe at the chapter's start timestamp

Output: videos/{video_id}/output/companion_guide.html
"""
import base64
import json
import re
import time
from pathlib import Path
from config import GEMINI_API_KEY, GEMINI_WRITE_MODEL, VIDEOS_DIR, OUTPUTS_DIR

MAX_RETRIES = 3


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def yt_thumbnail_facade(video_id: str, start: float) -> str:
    """
    YouTube thumbnail + play button overlay linking to the timestamped video.
    Avoids Error 153 (embed blocked for local file:// / computer:// origins).
    """
    thumb  = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    href   = yt_link_url(video_id, start)
    ts     = fmt_time(start)
    return f'''<a class="yt-facade" href="{href}" target="_blank" title="Watch at {ts} on YouTube">
          <img src="{thumb}" alt="Watch at {ts}" />
          <span class="play-btn">&#9654;</span>
        </a>'''


def yt_link_url(video_id: str, start: float) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s"


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_section(chapter: dict, video_id: str, idx: int, total: int) -> dict:
    """
    Generate prose for one chapter. Returns a dict with:
      {
        "title": str,
        "start": float,
        "prose": str,          # markdown-ish HTML prose
        "media_type": "screenshot" | "iframe",
        "frame_path": str | None,
      }
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    screen_type = chapter.get("screen_content_type", "none")
    frame_path = chapter.get("frame_path")

    prompt = f"""You are writing one section of a technical companion guide for a YouTube tutorial.
Style: clean technical prose, like Karpathy's lecture.md. No fluff, no "in this section we will".
Use ONLY these HTML tags: <p>, <pre><code class="language-python">, <ul><li>, <ol><li>, <strong>, <code>.
No headers — those are added separately.

Video section {idx+1} of {total}
Timestamp: {fmt_time(chapter['start'])} – {fmt_time(chapter['end'])}
Chapter title: {chapter['title']}
Screen content type: {screen_type}

Transcript:
\"\"\"{chapter['transcript']}\"\"\"

IMPORTANT — code extraction:
If the screenshot shows any code (Python, bash, config, etc.), transcribe it EXACTLY into a
<pre><code class="language-python"> block inside the prose. Do not paraphrase code. If multiple
code blocks are visible at different points, include all of them in order.

Media decision — use_screenshot:
- true  → only if code, terminal output, diagram, or non-trivial slide is clearly visible in the screenshot
- false → talking head, blurry/transitional frame, or content that reads fine without the image

Return JSON only (no markdown fences):
{{
  "prose": "<HTML prose — embed any visible code in <pre><code> blocks>",
  "use_screenshot": true | false,
  "reason": "<one sentence explaining the media decision>"
}}"""

    parts = [types.Part(text=prompt)]
    if frame_path and Path(frame_path).exists():
        parts.insert(0, types.Part(
            inline_data=types.Blob(mime_type="image/jpeg", data=encode_image(frame_path))
        ))

    for attempt in range(1, MAX_RETRIES + 1):
        response = client.models.generate_content(
            model=GEMINI_WRITE_MODEL,
            contents=types.Content(parts=parts),
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            result = json.loads(raw)
            break
        except json.JSONDecodeError as e:
            if attempt < MAX_RETRIES:
                print(f"    [JSON parse failed (attempt {attempt}): {e} — retrying...]")
                time.sleep(2)
            else:
                print(f"    [JSON parse failed after {MAX_RETRIES} attempts — using fallback]")
                result = _extract_json_fallback(raw)

    return {
        "title":       chapter["title"],
        "start":       chapter["start"],
        "prose":       result["prose"],
        "media_type":  "screenshot" if result.get("use_screenshot") else "iframe",
        "frame_path":  frame_path if result.get("use_screenshot") else None,
        "reason":      result.get("reason", ""),
    }


def _extract_json_fallback(raw: str) -> dict:
    """Best-effort extraction when Gemini returns malformed JSON."""
    # Try to grab prose between first "prose": " and the next top-level key
    prose_match = re.search(r'"prose"\s*:\s*"(.*?)",\s*"use_screenshot"', raw, re.DOTALL)
    prose = prose_match.group(1) if prose_match else "<p><em>[Section generation failed — see console.]</em></p>"
    use_ss = bool(re.search(r'"use_screenshot"\s*:\s*true', raw))
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
    return {
        "prose": prose,
        "use_screenshot": use_ss,
        "reason": reason_match.group(1) if reason_match else "fallback parse",
    }


def generate_guide(video_dir: Path, chapters: list, meta: dict, force: bool = False) -> Path:
    """
    Generate the full companion guide HTML.
    Saves to videos/{video_id}/output/companion_guide.html
    and copies to outputs/{slug}.html
    """
    output_dir = video_dir / "output"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "companion_guide.html"

    if out_path.exists() and not force:
        print("Guide already exists, skipping generation.")
        return out_path

    video_id = meta["video_id"]
    title    = meta["title"]
    url      = meta["url"]

    # Section-level cache so restarts resume from where they left off
    cache_dir = output_dir / "sections_cache"
    cache_dir.mkdir(exist_ok=True)

    print(f"Generating {len(chapters)} sections...")
    sections = []
    for i, chapter in enumerate(chapters):
        cache_file = cache_dir / f"section_{i:04d}.json"

        if cache_file.exists():
            section = json.loads(cache_file.read_text())
            print(f"  [{i+1}/{len(chapters)}] {chapter['title']} (cached)")
        else:
            print(f"  [{i+1}/{len(chapters)}] {chapter['title']}")
            section = generate_section(chapter, video_id, i, len(chapters))
            cache_file.write_text(json.dumps(section))
            print(f"    → {section['media_type']} ({section['reason']})")

        sections.append(section)

    html = _render_html(sections, title, url, video_id)
    out_path.write_text(html)
    print(f"Saved: {out_path}")

    # Copy to outputs/
    OUTPUTS_DIR.mkdir(exist_ok=True)
    slug = title.lower().replace(" ", "-")[:60]
    published = OUTPUTS_DIR / f"{slug}.html"
    published.write_text(html)
    print(f"Published: {published}")

    return out_path


def _render_html(sections: list, title: str, url: str, video_id: str) -> str:
    """Render sections into the final self-contained HTML."""

    def render_section(s: dict) -> str:
        ts_label = fmt_time(s["start"])
        ts_href  = yt_link_url(video_id, s["start"])

        if s["media_type"] == "screenshot" and s["frame_path"]:
            b64 = encode_image(s["frame_path"])
            media_html = f'''<figure class="screenshot">
          <img src="data:image/jpeg;base64,{b64}" alt="{s['title']}" />
          <figcaption>
            <a class="ts-link" href="{ts_href}" target="_blank">▶ {ts_label}</a>
          </figcaption>
        </figure>'''
        else:
            media_html = f'''<div class="iframe-wrap">
          {yt_thumbnail_facade(video_id, s["start"])}
        </div>'''

        return f'''<div class="section">
        <div class="section-header">
          <h2>{s['title']}</h2>
          <a class="ts" href="{ts_href}" target="_blank">{ts_label}</a>
        </div>
        {media_html}
        {s['prose']}
      </div>'''

    sections_html = "\n\n      <hr />\n\n      ".join(render_section(s) for s in sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title} — Companion Guide</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script>hljs.highlightAll();</script>
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;font-size:17px;line-height:1.75;color:#1a1a1a;background:#fafaf8}}
    .page{{max-width:760px;margin:0 auto;padding:60px 24px 120px}}
    .header{{border-bottom:2px solid #e8e8e4;padding-bottom:32px;margin-bottom:48px}}
    .badge{{display:inline-block;font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#666;background:#f0efe9;border:1px solid #ddd;border-radius:4px;padding:3px 10px;margin-bottom:16px}}
    h1{{font-size:2rem;font-weight:700;line-height:1.2;color:#111;margin-bottom:16px}}
    .meta{{font-size:14px;color:#888;display:flex;flex-wrap:wrap;gap:16px;align-items:center}}
    .yt-btn{{display:inline-flex;align-items:center;gap:6px;background:#ff0000;color:#fff;font-size:13px;font-weight:600;padding:6px 14px;border-radius:6px;text-decoration:none}}
    .yt-btn:hover{{background:#cc0000}}
    .section{{margin-bottom:64px}}
    .section-header{{display:flex;align-items:baseline;gap:12px;margin-bottom:20px}}
    h2{{font-size:1.35rem;font-weight:700;color:#111}}
    .ts{{flex-shrink:0;font-size:12px;font-weight:600;color:#3b82f6;background:#eff6ff;border:1px solid #bfdbfe;border-radius:4px;padding:2px 8px;text-decoration:none;white-space:nowrap}}
    .ts:hover{{background:#dbeafe}}
    p{{margin-bottom:16px}}
    .screenshot{{margin:24px 0;border:1px solid #e0e0d8;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
    .screenshot img{{width:100%;display:block}}
    .screenshot figcaption{{font-size:12.5px;padding:8px 12px;background:#f5f5f0;border-top:1px solid #e0e0d8}}
    .ts-link{{color:#3b82f6;text-decoration:none;font-weight:600}}
    .ts-link:hover{{text-decoration:underline}}
    .iframe-wrap{{margin:24px 0}}
    .yt-facade{{display:block;position:relative;border-radius:8px;overflow:hidden;border:1px solid #e0e0d8;box-shadow:0 2px 8px rgba(0,0,0,.06);aspect-ratio:16/9;text-decoration:none;background:#000}}
    .yt-facade img{{width:100%;height:100%;object-fit:cover;display:block;opacity:.92;transition:opacity .15s}}
    .yt-facade:hover img{{opacity:1}}
    .play-btn{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:64px;height:64px;background:rgba(0,0,0,.72);border-radius:50%;display:flex;align-items:center;justify-content:center;color:#fff;font-size:24px;padding-left:4px;transition:background .15s}}
    .yt-facade:hover .play-btn{{background:#ff0000}}
    pre{{margin:20px 0;border-radius:8px;border:1px solid #e0e0d8;overflow-x:auto;font-size:13.5px;line-height:1.6}}
    pre code.hljs{{padding:20px 22px;border-radius:8px}}
    code:not(pre>code){{background:#f0efe9;border:1px solid #ddd;border-radius:3px;padding:1px 5px;font-size:.88em;font-family:"SF Mono","Fira Code",Consolas,monospace;color:#c7254e}}
    ul{{padding-left:22px;margin-bottom:16px}}
    li{{margin-bottom:4px}}
    hr{{border:none;border-top:1px solid #e8e8e4;margin:48px 0}}
    .footer{{font-size:13px;color:#aaa;text-align:center;border-top:1px solid #e8e8e4;padding-top:24px}}
    .footer a{{color:#3b82f6;text-decoration:none}}
    .callout{{background:#fffbeb;border-left:3px solid #f59e0b;border-radius:0 6px 6px 0;padding:14px 18px;margin:20px 0;font-size:15px}}
  </style>
</head>
<body>
<div class="page">
  <div class="header">
    <div class="badge">Companion Guide</div>
    <h1>{title}</h1>
    <div class="meta">
      <span>Auto-generated with Gemini + Claude</span>
      <span>&middot;</span>
      <a class="yt-btn" href="{url}" target="_blank">&#9654; Watch on YouTube</a>
    </div>
  </div>

  {sections_html}

  <hr />
  <div class="footer">
    Auto-generated companion guide &nbsp;&middot;&nbsp;
    <a href="{url}" target="_blank">Watch the original video</a>
  </div>
</div>
</body>
</html>"""
