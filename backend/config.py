"""
FEK Processor - Configuration
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(BASE_DIR, "data")
INPUT_DIR       = os.path.join(DATA_DIR, "input")
OUTPUT_XML_DIR  = os.path.join(DATA_DIR, "output", "xml")
OUTPUT_HTML_DIR = os.path.join(DATA_DIR, "output", "html")
LAWS_DIR        = os.path.join(DATA_DIR, "laws")
REVIEW_DIR      = os.path.join(DATA_DIR, "review_queue")

# ── OCR ──────────────────────────────────────────────────────────────────────
TESSERACT_LANG      = "ell+eng"   # Greek + English
MIN_TEXT_CONFIDENCE = 0.6         # below this → treat page as scanned
MIN_CHARS_PER_PAGE  = 50          # below this → treat page as image

# ── Column detection ─────────────────────────────────────────────────────────
COLUMN_WHITE_RATIO  = 0.85        # % of vertical strip that must be white
COLUMN_SEARCH_BAND  = (0.40, 0.60)  # look for divider between 40–60 % of width

# ── Ollama model hierarchy ───────────────────────────────────────────────────
# Listed best-first; the manager picks the heaviest model that fits free RAM.
OLLAMA_MODELS = [
    {"name": "mistral:7b",   "min_free_ram_gb": 9.0},
    {"name": "qwen2.5:3b",   "min_free_ram_gb": 5.0},
    {"name": "phi3:mini",    "min_free_ram_gb": 3.0},
]
OLLAMA_BASE_URL   = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT    = 600   # seconds – large documents can take a while on CPU

OLLAMA_AUTO_START = True   # spawn `ollama serve` if not running
OLLAMA_AUTO_PULL  = True   # pull first model from OLLAMA_MODELS if none present

# Context window: 4 096 tokens ≈ ~3 000 words – enough for a typical FEK page.
# Raise to 8 192 if you have ≥ 16 GB RAM.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", 4096))

# Output tokens: the JSON schema response can easily be 1 000–2 000 tokens.
# 700 was too low and caused truncated / invalid JSON.
OLLAMA_MAX_OUTPUT_TOKENS = int(os.environ.get("OLLAMA_MAX_OUTPUT_TOKENS", 2048))

# ── Memory safety ────────────────────────────────────────────────────────────
RAM_DANGER_GB    = 2.5   # pause if free RAM drops below this (GB)

# ── Output ───────────────────────────────────────────────────────────────────
XML_ENCODING     = "utf-8"