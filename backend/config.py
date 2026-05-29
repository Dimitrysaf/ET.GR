"""
FEK Processor – Configuration
All tuneable values.  Override sensitive/env-specific ones via environment variables.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_XML_DIR = os.path.join(DATA_DIR, "output", "xml")
OUTPUT_HTML_DIR = os.path.join(DATA_DIR, "output", "html")
LAWS_DIR = os.path.join(DATA_DIR, "laws")
REVIEW_DIR = os.path.join(DATA_DIR, "review_queue")

# ── OCR ──────────────────────────────────────────────────────────────────────
TESSERACT_LANG = "ell+eng"  # Greek + English
MIN_TEXT_CONFIDENCE = 0.6  # below this → treat page as scanned
MIN_CHARS_PER_PAGE = 50  # below this → treat page as image

# ── Column detection ─────────────────────────────────────────────────────────
COLUMN_WHITE_RATIO = 0.85  # % of vertical strip that must be white
COLUMN_SEARCH_BAND = (0.40, 0.60)  # look for divider between 40–60 % of width

# ── Ollama model hierarchy ───────────────────────────────────────────────────
# Listed best-first; the manager picks the heaviest model that fits free RAM.
OLLAMA_MODELS = [
    {"name": "mistral:7b", "min_free_ram_gb": 9.0},
    {"name": "qwen2.5:3b", "min_free_ram_gb": 5.0},
    {"name": "phi3:mini", "min_free_ram_gb": 3.0},
]
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", 600))  # seconds

OLLAMA_AUTO_START = os.environ.get("OLLAMA_AUTO_START", "true").lower() != "false"
OLLAMA_AUTO_PULL = os.environ.get("OLLAMA_AUTO_PULL", "true").lower() != "false"

# Context window.
# ⚠️  IMPORTANT: 4096 is only ~3 000 words.  A ΦΕΚ page of dense legal text
# can easily exceed that.  The structurer.py chunking compensates, but larger
# values give better results when RAM allows:
#   • 4 096 → safe for all models on 8 GB RAM (default)
#   • 8 192 → better quality; needs ≥ 12 GB RAM for mistral:7b
#   • 16384 → best;           needs ≥ 20 GB RAM for mistral:7b
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", 4096))

# Output tokens.  Complex ΦΕΚ with many amendments can produce 1 500–2 500
# tokens of JSON output.  Too low → truncated JSON → parse errors.
OLLAMA_MAX_OUTPUT_TOKENS = int(os.environ.get("OLLAMA_MAX_OUTPUT_TOKENS", 2048))

# ── Memory safety ────────────────────────────────────────────────────────────
RAM_DANGER_GB = float(os.environ.get("RAM_DANGER_GB", 2.5))

# ── Output ───────────────────────────────────────────────────────────────────
XML_ENCODING = "utf-8"
