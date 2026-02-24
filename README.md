# postroll

> Turn any YouTube video into a deep companion guide. Watch once, read forever.

---

## Inspiration

[Andrej Karpathy](https://x.com/karpathy) put the idea perfectly:

> *"Fun LLM challenge that I'm thinking about: take my 2h13m tokenizer video and translate the video into the format of a book chapter (or a blog post) on tokenization. Something like: 1. Whisper the video 2. Chop up into segments of aligned images and text 3. Prompt engineer an LLM… More generally, a workflow like this could be applied to any input video and auto-generate 'companion guides' for various tutorials in a more readable, skimmable, searchable format. Feels tractable but non-trivial."*
>
> — [@karpathy](https://x.com/karpathy/status/1760740503614836917), Feb 22 2024

For that same video he also manually wrote [`lecture.md`](https://github.com/karpathy/minbpe/blob/master/lecture.md) — a dense companion guide covering every concept, code snippet, and paper reference. That's the quality bar.

**postroll builds the pipeline he described.** Give it any YouTube URL and it produces a structured HTML companion guide: timestamped sections, screenshots, and prose that goes deep — exact variable names, verbatim code, referenced papers.

---

## Demo

Input: `https://www.youtube.com/watch?v=zduSFxRajkE` (Karpathy's "Let's build the GPT tokenizer")

Output: 27 sections, ~485 words each, 100% video coverage, clickable timestamps.

---

## How it works

```
YouTube URL
    │
    ▼
 yt-dlp          →  downloads video + audio
    │
    ▼
 Gemini 2.5 Pro  →  transcribes + segments into chapters (dynamic scaling)
    │
    ▼
 ffmpeg          →  extracts per-section video clips + frame screenshots
    │
    ▼
 Gemini 2.5 Pro  →  writes prose for each section in parallel
 (sees the actual video clip, not just audio)
    │
    ▼
 HTML assembler  →  stitches sections into a single-file companion guide
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/tonykipkemboi/postroll.git
cd postroll

# 2. Install Python deps
pip install -r requirements.txt

# 3. Install ffmpeg (required for frame/clip extraction)
brew install ffmpeg        # macOS
sudo apt install ffmpeg    # Linux

# 4. Add your API key
cp .env.example .env
# edit .env and set GEMINI_API_KEY=...
```

---

## Usage

```bash
python -m agent_pipeline "https://www.youtube.com/watch?v=zduSFxRajkE"

# Force re-run (ignore cache)
python -m agent_pipeline "https://www.youtube.com/watch?v=zduSFxRajkE" --force
```

Output is saved to `outputs/<video-slug>.html`.

---

## Configuration

All knobs live in `agent_pipeline/config.py`:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_WRITE_MODEL` | `gemini-2.5-pro` | Model used for section prose |
| `SECTION_CONCURRENCY` | `5` | Max parallel section agents |
| `COVERAGE_THRESHOLD` | `0.80` | Min transcript coverage before warning |
| `USE_CLAUDE_FOR_SECTIONS` | `False` | Use Claude instead of Gemini for prose |

---

## Project structure

```
agent_run.py             # CLI entry point
agent_pipeline/
├── orchestrator.py      # Parallel section agents
├── config.py            # All config knobs
└── tools/
    ├── ingest.py        # yt-dlp download + metadata
    ├── transcribe.py    # Gemini transcription + chapter segmentation
    ├── frame.py         # ffmpeg frame extraction
    └── assemble.py      # HTML assembly
```

---

## Requirements

- Python 3.10+
- ffmpeg
- `GEMINI_API_KEY` (Gemini 2.5 Pro)
- `ANTHROPIC_API_KEY` (optional, for Claude sections)

---

## License

MIT
