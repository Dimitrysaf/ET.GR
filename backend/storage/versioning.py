"""Versioning for law texts by effective date."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Tuple

try:
    from backend.config import LAWS_DIR
except ImportError:
    from config import LAWS_DIR
from .organizer import slugify, write_text


def _law_dir(law_id: str) -> str:
    return os.path.join(LAWS_DIR, slugify(law_id))


def get_current_law_text(law_id: str) -> str:
    d = _law_dir(law_id)
    current = os.path.join(d, "current.txt")
    if not os.path.exists(current):
        return ""
    with open(current, "r", encoding="utf-8") as f:
        return f.read()


def snapshot_and_update(law_id: str, new_text: str, effective_date: Optional[str], diff: str = "") -> Tuple[str, str]:
    d = _law_dir(law_id)
    os.makedirs(d, exist_ok=True)

    current_path = os.path.join(d, "current.txt")
    date_label = effective_date or datetime.utcnow().strftime("%Y-%m-%d")
    version_path = os.path.join(d, f"version_{date_label}.txt")
    diff_path = os.path.join(d, f"diff_{date_label}.patch")

    old_text = ""
    if os.path.exists(current_path):
        with open(current_path, "r", encoding="utf-8") as f:
            old_text = f.read()

    if old_text:
        backup_path = os.path.join(d, f"previous_before_{date_label}.txt")
        write_text(backup_path, old_text)

    write_text(version_path, new_text)
    write_text(current_path, new_text)
    if diff:
        write_text(diff_path, diff)

    return current_path, version_path
