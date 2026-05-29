"""Human review queue manager."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, Any, List

try:
    from backend.config import REVIEW_DIR
except ImportError:
    from config import REVIEW_DIR


def ensure_review_dir() -> None:
    os.makedirs(REVIEW_DIR, exist_ok=True)


def enqueue_review(item: Dict[str, Any]) -> str:
    ensure_review_dir()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    doc_id = item.get("document_id", "unknown")
    # Αφαίρεση χαρακτήρων που δεν επιτρέπονται σε filenames (/, \, :, κ.λπ.)
    safe_id = re.sub(r'[/\\:*?"<>|]', "-", doc_id)
    filename = f"{timestamp}_{safe_id}.json"
    path = os.path.join(REVIEW_DIR, filename)

    payload = {
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **item,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path


def list_pending_reviews() -> List[Dict[str, Any]]:
    ensure_review_dir()
    out: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(REVIEW_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(REVIEW_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("status") == "pending":
            data["_file"] = path
            out.append(data)
    return out
