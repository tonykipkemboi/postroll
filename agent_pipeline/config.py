"""Agent pipeline configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
WORKSPACE  = BASE_DIR.parent
VIDEOS_DIR = WORKSPACE / "videos"
OUTPUTS_DIR = WORKSPACE / "outputs"

# ── API keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Add GEMINI_API_KEY=<your-key> to your .env file."
    )
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")  # optional; only needed for Claude sections

# ── Models ────────────────────────────────────────────────────────────────────
GEMINI_INGEST_MODEL   = "gemini-2.5-pro"          # long-video transcription
GEMINI_WRITE_MODEL    = "gemini-2.5-pro"          # section prose (best quality)

# ── Claude section agents (optional) ─────────────────────────────────────────
# To use Claude instead of Gemini for section prose:
#   1. Add ANTHROPIC_API_KEY=<your-key> to .env
#   2. In orchestrator.py, set USE_CLAUDE_FOR_SECTIONS = True
ANTHROPIC_SECTION_MODEL = "claude-haiku-4-5"
USE_CLAUDE_FOR_SECTIONS = False                   # flip to True once key is set

# ── Pipeline knobs ────────────────────────────────────────────────────────────
COVERAGE_THRESHOLD  = 0.80   # retry with file upload if URL coverage < 80%
SECTION_CONCURRENCY = 5      # max parallel section agents (clip uploads need headroom)
