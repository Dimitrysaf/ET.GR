"""AI structuring: OCR raw text → normalized document, XML, and HTML."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from lxml import etree as ET

from .model_manager import chat_json

SYSTEM_PROMPT = """
You normalize Greek legal gazette (ΦΕΚ) OCR text into strict JSON.
Respond with ONLY valid JSON – no markdown, no code fences, no explanation.

Output schema:
{
  "document_type": "fek|law|unknown",
  "title": "string",
  "document_id": "string",
  "publication_date": "YYYY-MM-DD or null",
  "sections": [{"kind": "heading|paragraph|note|citation", "text": "..."}],
  "signatures": ["..."],
  "metadata": {"source_hint": "...", "confidence": 0.0},
  "amendments": [{
    "target_law_id": "...",
    "operation": "replace|insert|delete|unknown",
    "article": "...",
    "old_text": "...",
    "new_text": "...",
    "confidence": 0.0,
    "question_for_human": "... or empty string"
  }]
}

Rules:
- Keep uncertain amendments with low confidence and a non-empty question_for_human.
- All string values must be valid JSON strings (escape special chars).
- The "amendments" array may be empty if no amendments are found.
- Output ONLY the JSON object, nothing else.
""".strip()


@dataclass
class StructuredOutput:
    data: Dict[str, Any]
    xml_text: str
    html_text: str
    needs_review: bool
    review_questions: List[str]


# ── JSON extraction ───────────────────────────────────────────────────────────


def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM output robustly:
    1. Strip leading/trailing whitespace.
    2. Strip markdown code fences (``` … ``` or ```json … ```).
    3. Extract the first complete {...} block if extra text surrounds it.
    4. Attempt JSON parse; raise ValueError with a snippet on failure.
    """
    text = raw.strip()

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # If the response contains extra text before/after the JSON object,
    # grab the first balanced { … } block.
    if not text.startswith("{"):
        m = re.search(r"\{", text)
        if m:
            text = text[m.start() :]

    if text.endswith("}") is False:
        # Find the last closing brace
        idx = text.rfind("}")
        if idx != -1:
            text = text[: idx + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        snippet = text[:300].replace("\n", " ")
        raise ValueError(f"LLM returned non-JSON output: {exc} | snippet: {snippet!r}")


# ── Validation ────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = [
    "document_type",
    "title",
    "document_id",
    "sections",
    "metadata",
    "amendments",
]


def _validate_llm_payload(data: Dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"LLM JSON missing required keys: {', '.join(missing)}")


# ── Core structuring ─────────────────────────────────────────────────────────


def structure_raw_text(
    raw_text: str,
    log_hook: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    prompt = (
        "Normalize the following OCR text according to the JSON schema. "
        "Mark uncertain items with low confidence and question_for_human.\n\n"
        f"OCR TEXT:\n{raw_text[:18000]}"
    )

    result = chat_json(prompt=prompt, system_prompt=SYSTEM_PROMPT, log_hook=log_hook)

    try:
        data = _extract_json(result["raw"])
    except (ValueError, json.JSONDecodeError) as exc:
        # Return a minimal valid stub so the pipeline can continue
        # and flag the document for human review.
        data = {
            "document_type": "unknown",
            "title": "PARSE ERROR – manual review required",
            "document_id": "parse-error",
            "publication_date": None,
            "sections": [{"kind": "note", "text": str(exc)}],
            "signatures": [],
            "metadata": {"confidence": 0.0, "parse_error": str(exc)},
            "amendments": [],
        }

    _validate_llm_payload(data)

    data.setdefault("metadata", {})
    data["metadata"].update(
        {
            "ai_mode": "llm",
            "model": result["model"],
            "model_reason": result["reason"],
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return data


# ── XML serialisation ─────────────────────────────────────────────────────────


def to_xml(data: Dict[str, Any]) -> str:
    root = ET.Element("document")

    for key in ["document_type", "title", "document_id", "publication_date"]:
        elem = ET.SubElement(root, key)
        elem.text = str(data.get(key) or "")

    sections_el = ET.SubElement(root, "sections")
    for sec in data.get("sections", []):
        sec_el = ET.SubElement(
            sections_el, "section", kind=sec.get("kind", "paragraph")
        )
        sec_el.text = sec.get("text", "")

    sigs_el = ET.SubElement(root, "signatures")
    for s in data.get("signatures", []):
        sig = ET.SubElement(sigs_el, "signature")
        sig.text = s

    meta_el = ET.SubElement(root, "metadata")
    for k, v in data.get("metadata", {}).items():
        item = ET.SubElement(meta_el, "item", key=str(k))
        item.text = str(v)

    amendments_el = ET.SubElement(root, "amendments")
    for a in data.get("amendments", []):
        am_el = ET.SubElement(amendments_el, "amendment")
        for k, v in a.items():
            child = ET.SubElement(am_el, k)
            child.text = "" if v is None else str(v)

    return ET.tostring(
        root, pretty_print=True, xml_declaration=True, encoding="utf-8"
    ).decode("utf-8")


# ── HTML serialisation ────────────────────────────────────────────────────────


def to_html(data: Dict[str, Any]) -> str:
    lines = ["<!doctype html>", "<html><head><meta charset='utf-8'></head><body>"]
    lines.append(f"<h1>{_esc(data.get('title', 'Untitled'))}</h1>")
    lines.append(f"<p><strong>ID:</strong> {_esc(data.get('document_id', ''))}</p>")
    lines.append(
        f"<p><strong>Type:</strong> {_esc(data.get('document_type', 'unknown'))}</p>"
    )
    if data.get("publication_date"):
        lines.append(f"<p><strong>Date:</strong> {_esc(data['publication_date'])}</p>")

    for sec in data.get("sections", []):
        kind = sec.get("kind", "paragraph")
        txt = _esc(sec.get("text", ""))
        if kind == "heading":
            lines.append(f"<h2>{txt}</h2>")
        elif kind == "note":
            lines.append(f"<blockquote>{txt}</blockquote>")
        elif kind == "citation":
            lines.append(f"<cite>{txt}</cite>")
        else:
            lines.append(f"<p>{txt}</p>")

    if data.get("signatures"):
        lines.append("<h3>Signatures</h3>")
        for s in data["signatures"]:
            lines.append(f"<p>{_esc(s)}</p>")

    lines.append("</body></html>")
    return "\n".join(lines)


def _esc(v: Any) -> str:
    return (
        str(v)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ── Public entry point ────────────────────────────────────────────────────────


def build_outputs(
    raw_text: str,
    log_hook: Optional[Callable[[str, str], None]] = None,
) -> StructuredOutput:
    data = structure_raw_text(raw_text, log_hook=log_hook)

    review_questions: List[str] = []
    for amend in data.get("amendments", []):
        if (amend.get("confidence") or 0) < 0.75 and amend.get("question_for_human"):
            review_questions.append(amend["question_for_human"])

    # Also flag if the whole document parse failed
    if data.get("metadata", {}).get("parse_error"):
        review_questions.append(
            "AI returned unparseable output – full manual review required."
        )

    needs_review = len(review_questions) > 0
    return StructuredOutput(
        data=data,
        xml_text=to_xml(data),
        html_text=to_html(data),
        needs_review=needs_review,
        review_questions=review_questions,
    )
