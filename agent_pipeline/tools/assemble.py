"""
Tool: assemble_html
Takes a list of section dicts and video metadata and renders the final HTML page.
"""
import html as _html
import math
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(secs: float) -> str:
    secs = int(secs)
    h, m = divmod(secs, 3600)
    m, s = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _yt_link(video_id: str, start: float) -> str:
    t = math.floor(start)
    return f"https://www.youtube.com/watch?v={video_id}&t={t}s"


def _thumbnail_facade(video_id: str, start: float) -> str:
    thumb = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    href  = _yt_link(video_id, start)
    ts    = _fmt_time(start)
    return (
        f'<a class="yt-facade" href="{href}" target="_blank" title="Watch at {ts} on YouTube">'
        f'<img src="{thumb}" alt="Watch at {ts}" />'
        f'<span class="play-btn">&#9654;</span>'
        f'</a>'
    )


CSS = """
/* ── reset + base ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 17px; line-height: 1.7; color: #1a1a1a;
  background: #fafaf8; padding: 0 16px 60px;
}
/* ── layout ──────────────────────────────────────────────────── */
.page-wrap { max-width: 740px; margin: 0 auto; }
header { padding: 40px 0 20px; border-bottom: 2px solid #e5e5e5; margin-bottom: 40px; }
header h1 { font-size: 2rem; font-weight: 700; }
header .meta { color: #666; font-size: 0.9rem; margin-top: 6px; }
/* ── sections ─────────────────────────────────────────────────── */
.section { margin-bottom: 52px; }
.section h2 {
  font-size: 1.35rem; font-weight: 700;
  margin-bottom: 14px;
  padding-bottom: 6px;
  border-bottom: 1px solid #e0e0e0;
}
.section p { margin: 0 0 14px; }
/* ── code blocks ──────────────────────────────────────────────── */
pre {
  background: #1e1e2e; color: #cdd6f4;
  border-radius: 8px; padding: 18px 20px;
  overflow-x: auto; margin: 20px 0;
  font-size: 0.85rem; line-height: 1.5;
}
code { font-family: "JetBrains Mono", "Fira Code", Consolas, monospace; }
p > code {
  background: #f0f0f0; color: #c7254e;
  padding: 2px 5px; border-radius: 4px; font-size: 0.88em;
}
/* ── frame images ─────────────────────────────────────────────── */
.frame-wrap {
  margin: 20px 0; border-radius: 10px;
  overflow: hidden; border: 1px solid #ddd;
  box-shadow: 0 2px 8px rgba(0,0,0,.06);
}
.frame-wrap img { width: 100%; display: block; }
/* ── youtube thumbnail facade ─────────────────────────────────── */
.yt-facade {
  display: block; position: relative;
  margin: 20px 0; border-radius: 10px;
  overflow: hidden; border: 1px solid #ddd;
  box-shadow: 0 2px 8px rgba(0,0,0,.06);
  aspect-ratio: 16/9;
  text-decoration: none;
}
.yt-facade img { width: 100%; height: 100%; object-fit: cover; display: block; }
.play-btn {
  position: absolute; top: 50%; left: 50%;
  transform: translate(-50%,-50%);
  width: 60px; height: 60px; border-radius: 50%;
  background: rgba(255,255,255,.9); color: #cc0000;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.4rem; transition: background .15s, transform .15s;
  padding-left: 4px;
}
.yt-facade:hover .play-btn { background: #cc0000; color: #fff; transform: translate(-50%,-50%) scale(1.08); }
/* ── toc ──────────────────────────────────────────────────────── */
.toc { background: #f4f4f0; border-radius: 10px; padding: 20px 24px; margin-bottom: 40px; }
.toc h3 { font-size: 1rem; margin-bottom: 12px; color: #555; text-transform: uppercase; letter-spacing: .05em; }
.toc ol { padding-left: 18px; }
.toc li { margin: 4px 0; }
.toc a { color: #1a56db; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
"""


def _section_html(section: dict, video_id: str) -> str:
    ch      = section["chapter"]
    idx     = section["index"]
    title   = _html.escape(ch["title"])   # XSS: escape before HTML interpolation
    start   = ch["start"]
    prose   = section.get("prose_html", "")
    frame   = section.get("frame_b64")
    use_fr  = section.get("use_frame", False)
    ts      = _fmt_time(start)

    # Always use the extracted frame if available; fall back to thumbnail only on failure
    if frame:
        media_html = (
            f'<div class="frame-wrap">'
            f'<a href="{_yt_link(video_id, start)}" target="_blank">'
            f'<img src="data:image/jpeg;base64,{frame}" alt="{title}" style="display:block;" />'
            f'</a>'
            f'</div>'
        )
    else:
        media_html = _thumbnail_facade(video_id, start)

    return (
        f'<article class="section" id="section-{idx:02d}">\n'
        f'  <h2>{title} <small style="font-weight:400;color:#888;font-size:.7em">— <a href="{_yt_link(video_id, start)}" target="_blank" style="color:#4a9eff;text-decoration:none;" onmouseover="this.style.textDecoration=\'underline\'" onmouseout="this.style.textDecoration=\'none\'">{ts}</a></small></h2>\n'
        f'  {media_html}\n'
        f'  <div class="prose">{prose}</div>\n'
        f'</article>\n'
    )


def _toc_html(chapters: list[dict], video_id: str) -> str:
    items = []
    for i, ch in enumerate(chapters):
        ts = _fmt_time(ch["start"])
        items.append(
            f'<li><a href="#section-{i:02d}">{_html.escape(ch["title"])}</a> '
            f'<span style="color:#999;font-size:.85em">{ts}</span></li>'
        )
    return (
        '<nav class="toc">'
        '<h3>Contents</h3>'
        '<ol>' + "\n".join(items) + "</ol>"
        "</nav>"
    )


# ── main tool ─────────────────────────────────────────────────────────────────

def assemble_html(sections: list[dict], meta: dict) -> str:
    """Build the complete HTML page from section dicts and video metadata."""
    video_id = meta["id"]
    title    = meta["title"]
    uploader = meta.get("uploader", "")
    duration = _fmt_time(meta.get("duration", 0))

    chapters = [s["chapter"] for s in sections]
    toc      = _toc_html(chapters, video_id)
    body     = "\n".join(_section_html(s, video_id) for s in sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <style>{CSS}</style>
</head>
<body>
<div class="page-wrap">
  <header>
    <h1>{title}</h1>
    <p class="meta">{uploader} &middot; {duration} &middot;
      <a href="https://www.youtube.com/watch?v={video_id}" target="_blank">Watch on YouTube</a>
    </p>
  </header>
  {toc}
  {body}
</div>
</body>
</html>"""
