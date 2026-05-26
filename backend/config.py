"""
FEK Processor - Configuration
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_DIR        = os.path.join(BASE_DIR, "data")
INPUT_DIR       = os.path.join(DATA_DIR, "input")
OUTPUT_XML_DIR  = os.path.join(DATA_DIR, "output", "xml")
OUTPUT_HTML_DIR = os.path.join(DATA_DIR, "output", "html")
LAWS_DIR        = os.path.join(DATA_DIR, "laws")
REVIEW_DIR      = os.path.join(DATA_DIR, "review_queue")

# OCR
TESSERACT_LANG      = "ell+eng"   # Greek + English
MIN_TEXT_CONFIDENCE = 0.6         # below this → scanned page
MIN_CHARS_PER_PAGE  = 50          # below this → treat as image

# Column detection
COLUMN_WHITE_RATIO  = 0.85        # % of vertical line that must be white
COLUMN_SEARCH_BAND  = (0.40, 0.60)  # look for divider between 40-60% of width

# Ollama model hierarchy (CPU-first for GTX 1070 without CUDA setup)
OLLAMA_MODELS = [
    {"name": "mistral:7b",    "min_free_ram_gb": 9.0},
    {"name": "qwen2.5:3b",   "min_free_ram_gb": 5.0},
    {"name": "phi3:mini",    "min_free_ram_gb": 3.0},
]
OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_TIMEOUT   = 300   # seconds per AI call

# Memory safety
RAM_DANGER_GB    = 2.5   # pause processing if free RAM drops below this

# Output
XML_ENCODING     = "utf-8"