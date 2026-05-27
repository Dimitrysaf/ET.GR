"""AI structuring for OCR raw text into normalized document, XML, and HTML."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Callable, Optional

from lxml import etree as ET

from .model_manager import chat_json


SYSTEM_PROMPT = """
You normalize Greek legal gazette OCR text into strict JSON.
Output schema:
{
  "document_type": "fek|law|unknown",
  "title": "string",
  "document_id": "string",
  "publication_date": "YYYY-MM-DD or null",
  "sections": [{"kind":"heading|paragraph|note|citation", "text":"..."}],
  "signatures": ["..."],
  "metadata": {"source_hint":"...", "confidence": 0.0},
  "amendments": [{
    "target_law_id":"...",
    "operation":"replace|insert|delete|unknown",
    "article":"...",
    "old_text":"...",
    "new_text":"...",
    "confidence": 0.0,
    "question_for_human":"... or empty"
  }]
}
Keep JSON valid. No markdown.
""".strip()


@dataclass
class StructuredOutput:
    data: Dict[str, Any]
    xml_text: str
    html_text: str
    needs_review: bool
    review_questions: List[str]


def _validate_llm_payload(data: Dict[str, Any]) -> None:
    required = ["document_type", "title", "document_id", "sections", "metadata", "amendments"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"LLM JSON missing required keys: {', '.join(missing)}")


def structure_raw_text(
    raw_text: str,
    log_hook: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    prompt = (
        "Normalize the following OCR text to the JSON schema. "
        "Keep uncertain items with low confidence and question_for_human.\n\n"
        f"OCR TEXT:\n{raw_text[:20000]}"
    )

    result = chat_json(prompt=prompt, system_prompt=SYSTEM_PROMPT, log_hook=log_hook)
    data = json.loads(result["raw"])
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


def to_xml(data: Dict[str, Any]) -> str:
    root = ET.Element("document")
    for key in ["document_type", "title", "document_id", "publication_date"]:
        elem = ET.SubElement(root, key)
        elem.text = str(data.get(key) or "")

    sections_el = ET.SubElement(root, "sections")
    for sec in data.get("sections", []):
        sec_el = ET.SubElement(sections_el, "section", kind=sec.get("kind", "paragraph"))
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
            item = ET.SubElement(am_el, k)
            item.text = "" if v is None else str(v)

    return ET.tostring(root, pretty_print=True, xml_declaration=True, encoding="utf-8").decode("utf-8")


def to_html(data: Dict[str, Any]) -> str:
    lines = ["<!doctype html>", "<html><body>"]
    lines.append(f"<h1>{_esc(data.get('title', 'Untitled'))}</h1>")
    lines.append(f"<p><strong>ID:</strong> {_esc(data.get('document_id', ''))}</p>")
    lines.append(f"<p><strong>Type:</strong> {_esc(data.get('document_type', 'unknown'))}</p>")
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


def _esc(v: str) -> str:
    return (
        str(v)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_outputs(
    raw_text: str,
    log_hook: Optional[Callable[[str, str], None]] = None,
) -> StructuredOutput:
    data = structure_raw_text(raw_text, log_hook=log_hook)
    review_questions = []
    for amend in data.get("amendments", []):
        if (amend.get("confidence") or 0) < 0.75 and amend.get("question_for_human"):
            review_questions.append(amend["question_for_human"])

    needs_review = len(review_questions) > 0
    return StructuredOutput(
        data=data,
        xml_text=to_xml(data),
        html_text=to_html(data),
        needs_review=needs_review,
        review_questions=review_questions,
    )
