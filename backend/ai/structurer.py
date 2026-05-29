"""AI structuring: OCR raw text → normalized document, XML, and HTML."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from lxml import etree as ET

from .model_manager import chat_json

# ── System prompt ─────────────────────────────────────────────────────────────
# Critical: small models (qwen2.5:3b, phi3:mini) tend to:
#   - Use camelCase field names instead of snake_case
#   - Add extra fields not in the schema
#   - Return incomplete JSON when output token limit is hit
# The one-shot example and explicit aliases address all three.

SYSTEM_PROMPT = """You are a specialized Greek legal document parser for ΦΕΚ (Greek Government Gazette).
Your goal is to transform OCR text into a high-fidelity structured JSON representation without losing ANY content.

REQUIRED OUTPUT SCHEMA:
{
  "document_type": "fek|law|presidential_decree|cabinet_act|ministerial_decision",
  "document_id": "ΦΕΚ Α' 1/2026",
  "publication_date": "YYYY-MM-DD",
  "act_number": "39",
  "act_year": "2025",
  "title": "Full descriptive title of the act",
  "preamble": [
    {"point": "1", "sub_points": ["α", "β"], "text": "..."}
  ],
  "body": [
    {
      "kind": "article",
      "number": "1",
      "title": "Title of article",
      "content": "Full text of article"
    },
    {
      "kind": "decision_block",
      "text": "Full text of decision"
    }
  ],
  "signatures": [
    {"title": "Ο Πρωθυπουργός", "name": "ΚΥΡΙΑΚΟΣ ΜΗΤΣΟΤΑΚΗΣ"},
    {"title": "Ο Υπουργός...", "name": "..."}
  ],
  "amendments": [
    {
      "target_law_id": "ν. 4622/2019",
      "operation": "replace|insert|delete",
      "article": "άρθρο 46 παρ. 6",
      "old_text": "...",
      "new_text": "..."
    }
  ],
  "metadata": {"confidence": 0.95}
}

STRICT RULES:
1. CONTENT FIDELITY: You must capture EVERY WORD from the source. Do not summarize preamble points (1, 2, 3...) or alphabetical sub-points (α, β, γ...).
2. PREAMBLE: Greek ΦΕΚ documents start with 'Έχοντας υπόψη:' followed by numbered points. Extract these EXHAUSTIVELY into the 'preamble' array.
3. BODY: For documents with articles, use kind='article'. For Cabinet Acts (Πράξεις Υπουργικού Συμβουλίου) which use 'αποφασίζει:', use kind='decision_block' to capture the decision text.
4. SIGNATURES: Find the list of signers at the end (e.g., 'Ο Πρωθυπουργός', 'Τα Μέλη...'). Extract both title and name.
5. TITLES: Act titles are usually between the 'ΠΡΑΞΕΙΣ ΥΠΟΥΡΓΙΚΟΥ ΣΥΜΒΟΥΛΙΟΥ' header and the preamble.
6. NO MARKDOWN: Output ONLY valid JSON.
""".strip()

# Minimal retry prompt - used when first attempt fails JSON parsing
RETRY_PROMPT = """Output ONLY valid JSON. No other text.

Extract from this Greek legal text:
- document_type (fek/law/unknown)
- title
- document_id (issue number like "ΦΕΚ Α' 1/2026")
- publication_date (YYYY-MM-DD or null)
- sections (array of {kind, text})
- signatures (array of strings)
- metadata ({confidence: 0.5})
- amendments (empty array [])

TEXT:
""".strip()


@dataclass
class StructuredOutput:
    data: Dict[str, Any]
    xml_text: str
    html_text: str
    needs_review: bool
    review_questions: List[str]


# ── Field name normalisation ──────────────────────────────────────────────────
# Maps common model mistakes (camelCase, alternate spellings) to canonical names.

_FIELD_ALIASES: Dict[str, str] = {
    "documentType": "document_type",
    "documentId": "document_id",
    "publicationDate": "publication_date",
    "datePublished": "publication_date",
    "date": "publication_date",
    "documentNumber": "document_id",
    "issueNumber": "document_id",
    "type": "document_type",
    "kind": "document_type",
    "actNumber": "act_number",
    "actYear": "act_year",
}


def _normalise_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Rename aliased top-level keys to canonical names."""
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        canonical = _FIELD_ALIASES.get(k, k)
        out[canonical] = v

    # Handle migration from old 'sections' to new 'body' if model uses old schema
    if "sections" in out and "body" not in out:
        out["body"] = []
        for sec in out["sections"]:
            if not isinstance(sec, dict): continue
            kind = sec.get("kind", "paragraph")
            text = sec.get("text", "")
            if kind == "heading" and "ΑΡΘΡΟ" in text.upper():
                # Try to extract article number
                m = re.search(r"ΑΡΘΡΟ\s+(\d+)", text, re.I)
                num = m.group(1) if m else "1"
                out["body"].append({"kind": "article", "number": num, "content": ""})
            elif out["body"] and out["body"][-1]["kind"] == "article" and not out["body"][-1].get("content"):
                out["body"][-1]["content"] = text
            else:
                out["body"].append({"kind": "decision_block", "text": text})

    return out


# ── JSON extraction ───────────────────────────────────────────────────────────


def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM output robustly:
    1. Strip markdown fences.
    2. Find the first balanced { … } block.
    3. Handle common truncation (missing closing braces).
    4. Normalise aliased field names.
    """
    text = raw.strip()

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    # Find start of JSON object
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output")
    text = text[start:]

    # Try to find balanced closing brace (handles truncated output)
    text = _balance_braces(text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # Last resort: try to salvage by cutting at the last valid comma
        salvaged = _salvage_truncated_json(text)
        if salvaged is not None:
            data = salvaged
        else:
            snippet = text[:400].replace("\n", " ")
            raise ValueError(
                f"LLM returned non-JSON output: {exc} | snippet: {snippet!r}"
            )

    if not isinstance(data, dict):
        raise ValueError(f"LLM returned JSON but not an object: {type(data)}")

    return _normalise_fields(data)


def _balance_braces(text: str) -> str:
    """
    If JSON is truncated mid-stream, close all open braces/brackets so
    json.loads has a chance of succeeding (values may be empty/partial).
    """
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            open_braces += 1
        elif ch == "}":
            open_braces = max(0, open_braces - 1)
        elif ch == "[":
            open_brackets += 1
        elif ch == "]":
            open_brackets = max(0, open_brackets - 1)

    # Close any open structures (may produce partial/empty values, but parseable)
    text = text.rstrip()
    # Remove trailing comma before closing (common with truncation)
    text = re.sub(r",\s*$", "", text)
    text += "]" * open_brackets + "}" * open_braces
    return text


def _salvage_truncated_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try progressively shorter versions of the text to find parseable JSON.
    Returns None if nothing works.
    """
    # Try cutting at the last complete key-value pair
    for pattern in [r",\s*\"[^\"]+\"\s*:\s*[^,\n]*$", r",\s*\{[^}]*$"]:
        candidate = re.sub(pattern, "", text.rstrip())
        candidate = _balance_braces(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


# ── Validation ────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = [
    "document_type",
    "title",
    "document_id",
    "preamble",
    "body",
    "metadata",
    "amendments",
]


def _validate_llm_payload(data: Dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"LLM JSON missing required keys: {', '.join(missing)}")


def _fill_missing_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in any missing required keys with safe defaults."""
    defaults: Dict[str, Any] = {
        "document_type": "unknown",
        "title": data.get("_author_hint", "Untitled"),
        "document_id": "unknown",
        "publication_date": None,
        "preamble": [],
        "body": [],
        "signatures": [],
        "metadata": {"confidence": 0.5},
        "amendments": [],
    }
    for k, v in defaults.items():
        data.setdefault(k, v)
    # Clean up internal hints
    data.pop("_author_hint", None)
    return data


# ── Text chunking ─────────────────────────────────────────────────────────────

# With OLLAMA_NUM_CTX=4096 we have room for ~2800 tokens of user text
# (system prompt ≈ 350 tokens, JSON schema ≈ 200 tokens, safety margin ≈ 500).
# 1 token ≈ 3.5 chars for Greek text → ~9800 chars per chunk.
# We use 4000 to ensure the structured JSON output (which is larger than raw text)
# fits within the model's output token limit (OLLAMA_MAX_OUTPUT_TOKENS).
_MAX_CHARS_PER_CHUNK = 5000


def _split_into_chunks(text: str) -> List[str]:
    """
    Split OCR text into page-sized chunks that fit within the context window.
    Tries to split on [PAGE N] markers first, then falls back to character count.
    """
    pages = re.split(r"(\[PAGE \d+\])", text)
    chunks: List[str] = []
    current = ""

    i = 0
    while i < len(pages):
        segment = pages[i]
        if i + 1 < len(pages) and re.match(r"\[PAGE \d+\]", pages[i + 1]):
            # Combine page marker with its content
            segment = (
                pages[i + 1] + pages[i + 2] if i + 2 < len(pages) else pages[i + 1]
            )
            i += 2
        else:
            i += 1

        if len(current) + len(segment) > _MAX_CHARS_PER_CHUNK and current:
            chunks.append(current.strip())
            current = segment
        else:
            current += segment

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text[:_MAX_CHARS_PER_CHUNK]]


def _merge_chunk_data(chunks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge structured data from multiple chunks into a single document.
    First chunk provides document-level metadata; all chunks contribute sections/amendments.
    """
    if not chunks_data:
        return {}
    if len(chunks_data) == 1:
        return chunks_data[0]

    merged = dict(chunks_data[0])
    for subsequent in chunks_data[1:]:
        # Extend preamble, body and amendments from subsequent chunks
        merged["preamble"] = merged.get("preamble", []) + subsequent.get("preamble", [])
        merged["body"] = merged.get("body", []) + subsequent.get("body", [])
        merged["amendments"] = merged.get("amendments", []) + subsequent.get(
            "amendments", []
        )
        merged["signatures"] = list(
            set(merged.get("signatures", [])) | set(subsequent.get("signatures", []))
        )
        # Use first non-null publication_date
        if not merged.get("publication_date") and subsequent.get("publication_date"):
            merged["publication_date"] = subsequent["publication_date"]
        # Upgrade confidence if we got better data
        existing_conf = merged.get("metadata", {}).get("confidence", 0)
        new_conf = subsequent.get("metadata", {}).get("confidence", 0)
        if new_conf > existing_conf:
            merged["metadata"]["confidence"] = new_conf

    return merged


# ── Core structuring ─────────────────────────────────────────────────────────


def _structure_chunk(
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    log_hook: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Structure a single text chunk, with one retry on JSON parse failure."""
    context_hint = (
        f" (chunk {chunk_index + 1}/{total_chunks})" if total_chunks > 1 else ""
    )

    prompt = (
        f"Normalize the following ΦΕΚ OCR text{context_hint} to the JSON schema. "
        "Mark uncertain amendments with low confidence and question_for_human.\n\n"
        f"OCR TEXT:\n{chunk_text}"
    )

    result = chat_json(prompt=prompt, system_prompt=SYSTEM_PROMPT, log_hook=log_hook)

    try:
        data = _extract_json(result["raw"])
        _fill_missing_keys(data)
        _validate_llm_payload(data)
        return data
    except (ValueError, json.JSONDecodeError) as first_exc:
        if log_hook:
            log_hook(
                "ai",
                f"chunk {chunk_index + 1} parse failed ({first_exc}), retrying with minimal prompt",
            )

        # Retry with a simpler prompt
        retry_prompt = RETRY_PROMPT + chunk_text[:4000]
        retry_result = chat_json(
            prompt=retry_prompt, system_prompt=SYSTEM_PROMPT, log_hook=log_hook
        )
        try:
            data = _extract_json(retry_result["raw"])
            _fill_missing_keys(data)
            _validate_llm_payload(data)
            data["metadata"]["retried"] = True
            return data
        except (ValueError, json.JSONDecodeError) as second_exc:
            # Both attempts failed – return a stub that keeps the pipeline running
            if log_hook:
                log_hook(
                    "ai", f"chunk {chunk_index + 1} retry also failed: {second_exc}"
                )
            return {
                "document_type": "unknown",
                "title": "PARSE ERROR – manual review required",
                "document_id": "parse-error",
                "publication_date": None,
                "sections": [
                    {"kind": "note", "text": f"[chunk {chunk_index + 1}] {second_exc}"}
                ],
                "signatures": [],
                "metadata": {"confidence": 0.0, "parse_error": str(second_exc)},
                "amendments": [],
            }


def structure_raw_text(
    raw_text: str,
    log_hook: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    chunks = _split_into_chunks(raw_text)
    total = len(chunks)

    if log_hook:
        log_hook(
            "ai", f"processing {total} chunk(s) from {len(raw_text)} chars of OCR text"
        )

    chunks_data: List[Dict[str, Any]] = []
    primary_model = None
    primary_reason = None

    for i, chunk in enumerate(chunks):
        data = _structure_chunk(chunk, i, total, log_hook=log_hook)
        if primary_model is None:
            primary_model = data.get("metadata", {}).get("model") or "unknown"
            primary_reason = data.get("metadata", {}).get("model_reason") or ""
        chunks_data.append(data)

    merged = _merge_chunk_data(chunks_data)
    _fill_missing_keys(merged)

    merged.setdefault("metadata", {})
    merged["metadata"].update(
        {
            "ai_mode": "llm",
            "model": primary_model,
            "model_reason": primary_reason,
            "chunks_processed": total,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return merged


# ── XML serialisation ─────────────────────────────────────────────────────────


def to_xml(data: Dict[str, Any]) -> str:
    root = ET.Element("document")

    for key in ["document_type", "title", "document_id", "publication_date", "act_number", "act_year"]:
        elem = ET.SubElement(root, key)
        elem.text = str(data.get(key) or "")

    preamble_el = ET.SubElement(root, "preamble")
    for p in data.get("preamble", []):
        p_el = ET.SubElement(preamble_el, "point", number=str(p.get("point", "")))
        p_el.text = p.get("text", "")
        if p.get("sub_points"):
            sps_el = ET.SubElement(p_el, "sub_points")
            for sp in p["sub_points"]:
                sp_el = ET.SubElement(sps_el, "sub_point")
                sp_el.text = str(sp)

    body_el = ET.SubElement(root, "body")
    for item in data.get("body", []):
        kind = item.get("kind", "unknown")
        item_el = ET.SubElement(body_el, "item", kind=kind)
        if kind == "article":
            num_el = ET.SubElement(item_el, "number")
            num_el.text = str(item.get("number", ""))
            title_el = ET.SubElement(item_el, "title")
            title_el.text = str(item.get("title", ""))
            cont_el = ET.SubElement(item_el, "content")
            cont_el.text = str(item.get("content", ""))
        else:
            txt_el = ET.SubElement(item_el, "text")
            txt_el.text = str(item.get("text", ""))

    sigs_el = ET.SubElement(root, "signatures")
    for s in data.get("signatures", []):
        sig = ET.SubElement(sigs_el, "signature")
        title_el = ET.SubElement(sig, "title")
        title_el.text = str(s.get("title", ""))
        name_el = ET.SubElement(sig, "name")
        name_el.text = str(s.get("name", ""))

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
    lines = ["<!doctype html>", "<html><head><meta charset='utf-8'>"]
    lines.append("<style>.diff-added { color: #00703c; background-color: #d7f1e6; } .diff-removed { color: #d4351c; background-color: #f9d6d2; } .fek-diff { font-family: monospace; white-space: pre-wrap; background-color: #f3f2f1; padding: 10px; border: 1px solid #b1b4b6; } .preamble-point { margin-left: 20px; } .sub-point { margin-left: 40px; font-style: italic; }</style>")
    lines.append("</head><body style='font-family: sans-serif; max-width: 900px; margin: 0 auto; padding: 40px;'>")

    # Header info
    lines.append(f"<p style='text-align: center; color: #666;'>{_esc(data.get('document_id', ''))} | {_esc(data.get('publication_date', ''))}</p>")
    lines.append(f"<h1 style='text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px;'>{_esc(data.get('title', 'Untitled'))}</h1>")

    # Preamble
    if data.get("preamble"):
        lines.append("<h2>Έχοντας υπόψη:</h2>")
        for p in data["preamble"]:
            text = _esc(p.get("text", ""))
            point = _esc(p.get("point", ""))
            lines.append(f"<div class='preamble-point'><p><strong>{point}.</strong> {text}</p></div>")
            if p.get("sub_points"):
                for sp in p["sub_points"]:
                    lines.append(f"<div class='sub-point'><p>{_esc(sp)}</p></div>")

    # Body
    if data.get("body"):
        for item in data["body"]:
            if item.get("kind") == "article":
                lines.append(f"<div style='margin-top: 30px;'>")
                lines.append(f"<h3 style='text-align: center;'>ΑΡΘΡΟ {_esc(item.get('number', ''))}</h3>")
                if item.get("title"):
                    lines.append(f"<h4 style='text-align: center;'>{_esc(item.get('title'))}</h4>")
                lines.append(f"<p>{_esc(item.get('content', ''))}</p>")
                lines.append(f"</div>")
            elif item.get("kind") == "decision_block":
                lines.append(f"<div style='margin-top: 30px; border-top: 1px solid #ccc; padding-top: 10px;'>")
                lines.append(f"<h3 style='text-align: center;'>ΑΠΟΦΑΣΙΖΕΙ</h3>")
                lines.append(f"<p>{_esc(item.get('text', ''))}</p>")
                lines.append(f"</div>")

    # Signatures
    if data.get("signatures"):
        lines.append("<div style='margin-top: 50px; display: flex; flex-wrap: wrap; justify-content: space-around;'>")
        for s in data["signatures"]:
            title = _esc(s.get("title", ""))
            name = _esc(s.get("name", ""))
            lines.append(f"<div style='text-align: center; margin: 20px;'><strong>{title}</strong><br>{name}</div>")
        lines.append("</div>")

    if data.get("amendments"):
        lines.append("<h2 style='margin-top: 60px; border-top: 2px dashed #005ea5; padding-top: 20px;'>Τροποποιήσεις (Diff View)</h2>")
        for amend in data["amendments"]:
            if amend.get("diff"):
                lines.append(f"<h3>Αλλαγή στο {amend.get('target_law_id')}</h3>")
                lines.append("<div class='fek-diff'>")
                for line in amend["diff"].splitlines():
                    cls = ""
                    if line.startswith("+"): cls = "diff-added"
                    elif line.startswith("-"): cls = "diff-removed"
                    lines.append(f"<div class='{cls}'>{_esc(line)}</div>")
                lines.append("</div>")

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
    # Check if raw_text is small enough for single chunk
    if len(raw_text) < 10000:
         if log_hook: log_hook("ai", "document small, using single chunk for maximum context")
         data = _structure_chunk(raw_text, 0, 1, log_hook=log_hook)
    else:
         data = structure_raw_text(raw_text, log_hook=log_hook)

    review_questions: List[str] = []
    for amend in data.get("amendments", []):
        if (amend.get("confidence", 0) or 0) < 0.75 and amend.get("question_for_human"):
            review_questions.append(amend["question_for_human"])

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
