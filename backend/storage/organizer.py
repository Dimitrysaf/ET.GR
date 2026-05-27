"""Folder and file organization helpers."""
from __future__ import annotations

import os
import re
from typing import Dict

try:
    from backend.config import OUTPUT_XML_DIR, OUTPUT_HTML_DIR, LAWS_DIR, REVIEW_DIR
except ImportError:
    from config import OUTPUT_XML_DIR, OUTPUT_HTML_DIR, LAWS_DIR, REVIEW_DIR


def ensure_dirs() -> None:
    for d in [OUTPUT_XML_DIR, OUTPUT_HTML_DIR, LAWS_DIR, REVIEW_DIR]:
        os.makedirs(d, exist_ok=True)


def slugify(value: str) -> str:
    value = (value or "untitled").strip().lower()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^a-z0-9\-_.\u0370-\u03FF\u1F00-\u1FFF]", "", value)
    return value or "untitled"


def output_paths(doc_id: str) -> Dict[str, str]:
    base = slugify(doc_id)
    return {
        "xml": os.path.join(OUTPUT_XML_DIR, f"{base}.xml"),
        "html": os.path.join(OUTPUT_HTML_DIR, f"{base}.html"),
    }


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
