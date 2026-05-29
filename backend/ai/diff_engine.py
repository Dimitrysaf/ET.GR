"""Apply FEK amendments to law text with version snapshotting semantics."""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple


@dataclass
class AmendmentResult:
    changed: bool
    low_confidence: bool
    notes: List[str]
    new_law_text: str
    diff: str = ""


def apply_amendments(current_law_text: str, amendments: List[Dict[str, Any]]) -> AmendmentResult:
    text = current_law_text
    changed = False
    low_confidence = False
    notes: List[str] = []

    for idx, am in enumerate(amendments, start=1):
        op = (am.get("operation") or "unknown").lower()
        old_text = am.get("old_text") or ""
        new_text = am.get("new_text") or ""
        confidence = float(am.get("confidence") or 0)

        if confidence < 0.75:
            low_confidence = True
            notes.append(f"amendment #{idx} low confidence ({confidence:.2f})")

        if op == "replace":
            if old_text and old_text in text:
                text = text.replace(old_text, new_text, 1)
                changed = True
            else:
                notes.append(f"replace failed in amendment #{idx}: old_text not found")
                low_confidence = True
        elif op == "insert":
            anchor = old_text.strip()
            if anchor and anchor in text:
                text = text.replace(anchor, anchor + "\n" + new_text, 1)
            else:
                text += "\n\n" + new_text
            changed = True
        elif op == "delete":
            if old_text and old_text in text:
                text = text.replace(old_text, "", 1)
                changed = True
            else:
                notes.append(f"delete failed in amendment #{idx}: old_text not found")
                low_confidence = True
        else:
            notes.append(f"unknown operation in amendment #{idx}")
            low_confidence = True

    diff = ""
    if changed:
        diff_lines = difflib.unified_diff(
            current_law_text.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile='current',
            tofile='updated'
        )
        diff = "".join(diff_lines)

    return AmendmentResult(
        changed=changed,
        low_confidence=low_confidence,
        notes=notes,
        new_law_text=text,
        diff=diff
    )
