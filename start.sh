#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────
# FEK Processor – one-shot startup
#   1. Creates / activates a venv
#   2. Installs / updates Python deps
#   3. Checks for Tesseract
#   4. Starts the FastAPI server (Ollama auto-start is handled by the app)
# ────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-}"   # set RELOAD=--reload for dev

# ── 1. Virtualenv ────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "→ Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "→ Installing / updating Python dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

# ── 2. Tesseract check ───────────────────────────────────────
if ! command -v tesseract &>/dev/null; then
    echo ""
    echo "⚠  Tesseract is not installed.  OCR of scanned pages will be skipped."
    echo "   Install with:"
    echo "     Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-ell"
    echo "     macOS:         brew install tesseract tesseract-lang"
    echo ""
fi

# ── 3. Create required data directories ─────────────────────
for d in data/input data/output/xml data/output/html data/laws data/review_queue; do
    mkdir -p "$SCRIPT_DIR/$d"
done

# ── 4. Start server ──────────────────────────────────────────
echo ""
echo "✓  Starting FEK Processor on http://${HOST}:${PORT}"
echo "   Open your browser at: http://localhost:${PORT}"
echo ""

cd "$SCRIPT_DIR"
exec uvicorn backend.main:app --host "$HOST" --port "$PORT" ${RELOAD}