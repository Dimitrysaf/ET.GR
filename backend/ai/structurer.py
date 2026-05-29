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

SYSTEM_PROMPT = """You are an expert Greek legal document parser specializing in ΦΕΚ (Efimerides tis Kyverniseos - Official Government Gazette).

CRITICAL: Output ONLY a single valid JSON object with NO markdown, NO code fences, NO explanation. Only the JSON.

YOUR TASK: Extract the document into a SEMANTICALLY STRUCTURED format that preserves legal meaning and hierarchies.

═══════════════════════════════════════════════════════════════════════════════
BASIC METADATA (all documents):
═══════════════════════════════════════════════════════════════════════════════

{
  "document_type": "cabinet_act" or "law" or "presidential_decree" or "fek",
  "title": "FULL descriptive title - NEVER shorten, extract complete title",
  "document_id": "ΦΕΚ Α' 1/2026 or ν. 4622/2019 (look for Αρ. Φύλλου, Τεύχος, law number)",
  "publication_date": "YYYY-MM-DD or null",
  
═══════════════════════════════════════════════════════════════════════════════
IF DOCUMENT TYPE = "cabinet_act" (Πράξη Υπουργικού Συμβουλίου):
═══════════════════════════════════════════════════════════════════════════════

  "document_structure": "cabinet_act",
  "action_info": {
    "number": "39" (extract from 'Πράξη Νο 39'),
    "date": "2025-12-23" (format as YYYY-MM-DD),
    "title": "Full title of the act"
  },
  "organ": "ΤΟ ΥΠΟΥΡΓΙΚΟ ΣΥΜΒΟΥΛΙΟ",
  
  "aitiologia": {
    "intro": "Έχοντας υπόψη:",
    "items": [
      {
        "number": 1,
        "intro": "Tις διατάξεις:",
        "subitems": [
          {"letter": "α", "text": "Των παρ. 1 και 2 του άρθρου 45..."},
          {"letter": "β", "text": "Του άρθρου 90..."},
          ...
        ]
      },
      {
        "number": 2,
        "text": "Την ανάγκη αύξησης κατά επτά (7) των θέσεων..."
      },
      ...
    ]
  },
  
  "decision": {
    "intro": "αποφασίζει:",
    "articles": [
      {
        "number": 1,
        "title": null or "article title if exists",
        "text": "Αυξάνονται κατά επτά (7) οι θέσεις..."
      },
      {
        "number": 2,
        "text": "..."
      }
    ],
    "publication_instruction": "Η παρούσα Πράξη να δημοσιευθεί στην Εφημερίδα της Κυβέρνησης."
  },
  
  "signatures": {
    "prothypourgos": "ΚΥΡΙΑΚΟΣ ΜΗΤΣΟΤΑΚΗΣ",
    "members": [
      "ΚΩΝΣΤΑΝΤΙΝΟΣ ΧΑΤΖΗΔΑΚΗΣ",
      "ΚΥΡΙΑΚΟΣ ΠΙΕΡΡΑΚΑΚΗΣ",
      ...
    ]
  },
  
  "contact_info": {
    "address": "Καποδιστρίου 34, Τ.Κ. 104 32 Αθήνα" or null,
    "phone": "210 5279000" or null,
    "url": "https://eservices.et.gr" or null
  },

═══════════════════════════════════════════════════════════════════════════════
IF DOCUMENT TYPE = "law" or "presidential_decree" or "fek" (fallback):
═══════════════════════════════════════════════════════════════════════════════

  "sections": [
    {"kind": "heading", "text": "ΑΡΘΡΟ 1"},
    {"kind": "paragraph", "text": "body text"},
    ...
  ],
  "signatures": ["ΥΠΟΥΡΓΟΣ name"],

═══════════════════════════════════════════════════════════════════════════════
COMMON FIELDS (all documents):
═══════════════════════════════════════════════════════════════════════════════

  "metadata": {
    "confidence": 0.9 (float 0.0-1.0),
    "source_hint": "gazette issue info if applicable",
    "chunks_processed": 1
  },
  
  "amendments": [
    {
      "target_law_id": "ν. 4622/2019",
      "operation": "replace" or "add" or "repeal",
      "article": "άρθρο 46 παρ. 6",
      "old_text": "text being replaced",
      "new_text": "replacement text",
      "confidence": 0.9
    }
  ] (empty [] if no amendments)
}

═══════════════════════════════════════════════════════════════════════════════
CRITICAL RULES:
═══════════════════════════════════════════════════════════════════════════════

1. EXTRACT EVERYTHING: Do NOT summarize, abbreviate, or omit text. Every sentence MUST be included.
2. NUMBERED ITEMS: When you see "1. text", "2. text", or "(α) text", "(β) text", preserve the numbers/letters.
3. DATES: Always convert to YYYY-MM-DD. Handle Greek day names like "8 Ιανουαρίου 2026" → "2026-01-08".
4. FIELD NAMES: MUST be snake_case EXACTLY as shown. Never use camelCase.
5. CABINET ACT PRIORITY: If you detect "Πράξη Υπουργικού Συμβουλίου" OR "ΤΟ ΥΠΟΥΡΓΙΚΟ ΣΥΜΒΟΥΛΙΟ", use the cabinet_act structure EXCLUSIVELY.
6. AITIOLOGIA STRUCTURE: Split into numbered items with sub-items (α, β, γ, etc.) where they exist.
7. DECISION STRUCTURE: Extract articles separately with their text (do NOT mix into single block).
8. SIGNATURES: Extract PM name separately from cabinet members. List all member names.
9. CONFIDENCE: If uncertain about any extraction, mark confidence <0.75 and add question_for_human.
10. No field names with typos (e.g., "Apofasistico" MUST be split per JSON rules, not renamed).
""".strip()

# Minimal retry prompt - used when first attempt fails JSON parsing
RETRY_PROMPT = """Output ONLY valid JSON. No other text. Extract from this Greek legal text.

For Cabinet Acts (Πράξη Υπουργικού Συμβουλίου): Use this structure:
{
  "document_type": "cabinet_act",
  "title": "full title",
  "document_id": "ΦΕΚ Α' 1/2026",
  "publication_date": "YYYY-MM-DD",
  "document_structure": "cabinet_act",
  "action_info": {"number": "39", "date": "2025-12-23", "title": "..."},
  "organ": "ΤΟ ΥΠΟΥΡΓΙΚΟ ΣΥΜΒΟΥΛΙΟ",
  "aitiologia": {"intro": "Έχοντας υπόψη:", "items": [...]},
  "decision": {"intro": "αποφασίζει:", "articles": [...]},
  "signatures": {"prothypourgos": "...", "members": [...]},
  "contact_info": {"address": "...", "phone": "...", "url": "..."},
  "metadata": {"confidence": 0.9},
  "amendments": []
}

For other documents: Use generic structure with "sections" array.

ABSOLUTELY ALL TEXT MUST BE EXTRACTED - NO OMISSIONS.
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
    "author": "_author_hint",  # not in schema, capture for title fallback
    "issuer": "_author_hint",
    # Cabinet act structure aliases
    "actionInfo": "action_info",
    "action_Info": "action_info",
    "documentStructure": "document_structure",
    "contactInfo": "contact_info",
    "contact_Info": "contact_info",
}

_SECTION_KIND_ALIASES: Dict[str, str] = {
    "header": "heading",
    "title": "heading",
    "body": "paragraph",
    "text": "paragraph",
    "footnote": "note",
    "ref": "citation",
    "reference": "citation",
}


def _normalise_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Rename aliased top-level keys to canonical names."""
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        canonical = _FIELD_ALIASES.get(k, k)
        out[canonical] = v

    # Ensure sections have valid 'kind' values
    for sec in out.get("sections", []):
        if isinstance(sec, dict):
            sec["kind"] = _SECTION_KIND_ALIASES.get(
                sec.get("kind", "paragraph"), sec.get("kind", "paragraph")
            )
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
    "sections",
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
        "sections": [],
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
# We use 6000 as a balance between context size and output token limit.
_MAX_CHARS_PER_CHUNK = 6000


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
        # Extend sections and amendments from subsequent chunks
        merged["sections"] = merged.get("sections", []) + subsequent.get("sections", [])
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


# ── Greek ΦΕΚ XML helpers ─────────────────────────────────────────────────────

greek_months = {
    "Ιανουαρίου": 1,
    "Φεβρουαρίου": 2,
    "Μαρτίου": 3,
    "Απριλίου": 4,
    "Μαΐου": 5,
    "Ιουνίου": 6,
    "Ιουλίου": 7,
    "Αυγούστου": 8,
    "Σεπτεμβρίου": 9,
    "Οκτωβρίου": 10,
    "Νοεμβρίου": 11,
    "Δεκεμβρίου": 12,
}


def _normalize_raw_text(raw_text: str) -> str:
    text = raw_text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _format_greek_date(match: str) -> Optional[str]:
    match = match.strip()
    if m := re.search(r"(\d{1,2})\s+([Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώ]+)\s+(\d{4})", match):
        day = int(m.group(1))
        month_name = m.group(2)
        year = int(m.group(3))
        month = greek_months.get(month_name)
        if month:
            return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _format_dot_date(match: str) -> Optional[str]:
    match = match.strip()
    if m := re.search(r"(\d{2})\.(\d{2})\.(\d{4})", match):
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None


def _extract_one(pattern: str, text: str, flags: int = re.I) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _split_alpha_list(text: str) -> List[Dict[str, str]]:
    parts = re.split(r"\s*([α-ωΑ-Ω])\)\s*", text)
    items: List[Dict[str, str]] = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            letter, content = parts[i], parts[i + 1].strip()
            if content:
                items.append({"grammato": letter, "text": content})
    return items


def _split_numbered_items(text: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for match in re.finditer(r"(\d+)\.\s*(.*?)(?=(?:\d+\.|$))", text, re.S):
        items.append({"arithmos": match.group(1), "text": match.group(2).strip()})
    return items


def _parse_signatures(text: str) -> Dict[str, Any]:
    result = {"prothypourgos": None, "members": []}
    if "Ο Πρωθυπουργός" in text and "Τα Μέλη του Υπουργικού Συμβουλίου" in text:
        result["prothypourgos"] = _extract_one(
            r"Ο\s*Πρωθυπουργός\s*(.*?)\s*Τα\s*Μέλη\s*του\s*Υπουργικού\s*Συμβουλίου",
            text,
            flags=re.S | re.I,
        )
        members_text = _extract_one(
            r"Τα\s*Μέλη\s*του\s*Υπουργικού\s*Συμβουλίου\s*(.*?)(?:Καποδιστρίου|Τηλ\.|https?://|$)",
            text,
            flags=re.S | re.I,
        )
        if members_text:
            result["members"] = [m.strip() for m in re.split(r",\s*", members_text) if m.strip()]
    else:
        if "Ο Πρωθυπουργός" in text:
            result["prothypourgos"] = _extract_one(
                r"Ο\s*Πρωθυπουργός\s*([Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώ\s\-]+)",
                text,
                flags=re.I,
            )
        if "Τα Μέλη του Υπουργικού Συμβουλίου" in text:
            members_text = _extract_one(
                r"Τα\s*Μέλη\s*του\s*Υπουργικού\s*Συμβουλίου\s*(.*)$",
                text,
                flags=re.S | re.I,
            )
            if members_text:
                result["members"] = [m.strip() for m in re.split(r",\s*", members_text) if m.strip()]
    return result


def _parse_contact_info(text: str) -> Dict[str, str]:
    return {
        "address": _extract_one(r"(Καποδιστρίου[^Τ]*?)\s*(?:Τηλ\.|https?://|$)", text, flags=re.S) or "",
        "phone": _extract_one(r"Τηλ\.\s*[:\s]*([0-9\s\-]+)", text, flags=re.I) or "",
        "url": _extract_one(r"(https?://\S+)", text, flags=re.I) or "",
    }


def _build_fek_cabinet_act_xml(data: Dict[str, Any]) -> str:
    """
    Build proper FEK XML from a cabinet_act JSON structure.
    Expects data with: action_info, aitiologia, decision, signatures, contact_info.
    """
    # Define namespaces
    XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
    nsmap = {
        None: None,  # Default namespace
        "xsi": XSI_NS,
    }
    
    root = ET.Element("EfimeridaKyverniseos", nsmap=nsmap)
    root.set("{" + XSI_NS + "}noNamespaceSchemaLocation", "fek.xsd")

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata_el = ET.SubElement(root, "Metadata")
    
    title_el = ET.SubElement(metadata_el, "Titlos")
    title_el.text = "Εφημερίδα της Κυβερνήσεως της Ελληνικής Δημοκρατίας"
    
    # Extract issue info from document_id (e.g., "ΦΕΚ Α' 1/2026" → Teychos="Α'", ArithmosFyllou="1")
    doc_id = data.get("document_id", "")
    issue_match = re.search(r"Α'|Β'|Γ'|Δ'", doc_id)
    issue = issue_match.group(0) if issue_match else "Α'"
    
    num_match = re.search(r"/(\d+)$", doc_id)
    issue_num = num_match.group(1) if num_match else "1"
    
    teychos_el = ET.SubElement(metadata_el, "Teychos")
    teychos_el.text = issue
    
    arithmos_el = ET.SubElement(metadata_el, "ArithmosFyllou")
    arithmos_el.text = issue_num
    
    hmerominia_el = ET.SubElement(metadata_el, "Hmerominia")
    hmerominia_el.text = data.get("publication_date", "")

    # ── Periexomena (Contents) ────────────────────────────────────────────────
    periexomena = ET.SubElement(root, "Periexomena")
    kefalaio = ET.SubElement(periexomena, "Kefalaio")
    kefalaio.set("titlos", "ΠΡΑΞΕΙΣ ΥΠΟΥΡΓΙΚΟΥ ΣΥΜΒΟΥΛΙΟΥ")

    # ── Praxi (Act) ───────────────────────────────────────────────────────────
    praxi = ET.SubElement(kefalaio, "Praxi")
    
    action_info = data.get("action_info", {})
    if action_info.get("number"):
        praxi.set("arithmos", str(action_info["number"]))
    if action_info.get("date"):
        praxi.set("hmerominia", str(action_info["date"]))

    # Title
    praxi_title = ET.SubElement(praxi, "Titlos")
    praxi_title.text = action_info.get("title") or data.get("title", "")

    # Organ
    organ_el = ET.SubElement(praxi, "Organ")
    organ_el.text = data.get("organ", "ΤΟ ΥΠΟΥΡΓΙΚΟ ΣΥΜΒΟΥΛΙΟ")

    # ── Aitiologia (Reasoning) ────────────────────────────────────────────────
    aitiologia = data.get("aitiologia", {})
    if aitiologia:
        aitiologia_el = ET.SubElement(praxi, "AitiolologikoMeros")
        
        if aitiologia.get("intro"):
            eisagogi_el = ET.SubElement(aitiologia_el, "Eisagogi")
            eisagogi_el.text = aitiologia["intro"]
        
        for item in aitiologia.get("items", []):
            stoicheio_el = ET.SubElement(aitiologia_el, "Stoicheio")
            stoicheio_el.set("arithmos", str(item.get("number", "")))
            
            if item.get("intro"):
                intro_el = ET.SubElement(stoicheio_el, "Eisagogi")
                intro_el.text = item["intro"]
            
            # Handle subitems (α, β, γ, etc.)
            if item.get("subitems"):
                lista_el = ET.SubElement(stoicheio_el, "Lista")
                for subitem in item["subitems"]:
                    peripton_el = ET.SubElement(lista_el, "Peripton")
                    peripton_el.set("grammato", str(subitem.get("letter", "")))
                    peripton_el.text = subitem.get("text", "")
            elif item.get("text"):
                stoicheio_el.text = item["text"]

    # ── Decision (Apofasistiko Meros) ─────────────────────────────────────────
    decision = data.get("decision", {})
    if decision:
        apofasi_el = ET.SubElement(praxi, "ApoфasistikоMeros")
        
        if decision.get("intro"):
            eisagogi_dec = ET.SubElement(apofasi_el, "Eisagogi")
            eisagogi_dec.text = decision["intro"]
        
        for article in decision.get("articles", []):
            arthro_el = ET.SubElement(apofasi_el, "Arthro")
            if article.get("number"):
                arthro_el.set("arithmos", str(article["number"]))
            if article.get("title"):
                arthro_title = ET.SubElement(arthro_el, "Titlos")
                arthro_title.text = article["title"]
            if article.get("text"):
                arthro_text = ET.SubElement(arthro_el, "Keimeno")
                arthro_text.text = article["text"]
        
        if decision.get("publication_instruction"):
            diatagh_el = ET.SubElement(apofasi_el, "DiataghDimosiefseos")
            diatagh_el.text = decision["publication_instruction"]

    # ── Signatures (Ypografes) ────────────────────────────────────────────────
    signatures = data.get("signatures", {})
    if signatures:
        ypografes_el = ET.SubElement(praxi, "Ypografes")
        
        if signatures.get("prothypourgos"):
            pro_el = ET.SubElement(ypografes_el, "Prothypourgos")
            onoma_el = ET.SubElement(pro_el, "Onoma")
            onoma_el.text = signatures["prothypourgos"]
        
        if signatures.get("members"):
            members_el = ET.SubElement(ypografes_el, "MeliYpourgikoySymbouliou")
            for member in signatures["members"]:
                melos_el = ET.SubElement(members_el, "Melos")
                melos_el.text = member

    # ── Contact Info (Epikoinonia) ────────────────────────────────────────────
    contact_info = data.get("contact_info", {})
    if any([contact_info.get("address"), contact_info.get("phone"), contact_info.get("url")]):
        epikoinonia_el = ET.SubElement(root, "Epikoinonia")
        
        if contact_info.get("address"):
            dieuth_el = ET.SubElement(epikoinonia_el, "Dieuthinsi")
            dieuth_el.text = contact_info["address"]
        
        if contact_info.get("phone"):
            tilel_el = ET.SubElement(epikoinonia_el, "Tilefono")
            tilel_el.text = contact_info["phone"]
        
        if contact_info.get("url"):
            url_el = ET.SubElement(epikoinonia_el, "HlektronikaDieuthinsi")
            url_el.text = contact_info["url"]

    return ET.tostring(
        root, pretty_print=True, xml_declaration=True, encoding="utf-8"
    ).decode("utf-8")


def _build_efimerida_xml(data: Dict[str, Any], raw_text: str) -> str:
    text = _normalize_raw_text(raw_text)
    issue = _extract_one(r"ΤΕΥΧΟΣ\s*([Α-Ωα-ω'’]+)", text, flags=re.I) or _extract_one(r"Τεύχος\s*([Α-Ωα-ω'’]+)", text, flags=re.I) or ""
    issue_num = _extract_one(r"Αρ\.\s*Φύλλου\s*(\d+)", text, flags=re.I) or ""
    pub_date_text = _extract_one(r"(\d{1,2}\s+[Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώ]+\s+\d{4})", text, flags=re.I)
    if pub_date_text:
        pub_date = _format_greek_date(pub_date_text) or ""
    else:
        pub_date = _extract_one(r"(\d{2}\.\d{2}\.\d{4})", text, flags=re.I) or ""
        pub_date = _format_dot_date(pub_date) or pub_date

    action_match = re.search(
        r"Πράξη\s+(\d+)\s+της\s+(\d{2}-\d{2}-\d{4})\s*(.*?)\s+ΤΟ\s+ΥΠΟΥΡΓΙΚΟ\s+ΣΥΜΒΟΥΛΙΟ",
        text,
        flags=re.I | re.S,
    )
    action_number = action_match.group(1) if action_match else ""
    action_date = action_match.group(2) if action_match else ""
    action_title = action_match.group(3).strip() if action_match else data.get("title", "")
    organ = "ΤΟ ΥΠΟΥΡΓΙΚΟ ΣΥΜΒΟΥΛΙΟ" if re.search(r"ΤΟ\s+ΥΠΟΥΡΓΙΚΟ\s+ΣΥΜΒΟΥΛΙΟ", text, flags=re.I) else ""

    aitiologia = _extract_one(
        r"Έχοντας υπόψη:\s*(.*?)(?:αποφασίζει:|αποφασίζει|Αποφασίζει:|Αποφασίζει)",
        text,
        flags=re.I | re.S,
    )
    decision_text = _extract_one(
        r"(?:αποφασίζει:|αποφασίζει|Αποφασίζει:|Αποφασίζει)\s*(.*?)(?:Ο\s*Πρωθυπουργός|Τα\s*Μέλη\s*του\s*Υπουργικού\s*Συμβουλίου|Καποδιστρίου|Τηλ\.|https?://|$)",
        text,
        flags=re.I | re.S,
    )

    signature_info = _parse_signatures(text)
    contact_info = _parse_contact_info(text)

    root = ET.Element("EfimeridaKyverniseos")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "fek.xsd")

    metadata_el = ET.SubElement(root, "Metadata")
    title_el = ET.SubElement(metadata_el, "Titlos")
    title_el.text = "Εφημερίδα της Κυβερνήσεως της Ελληνικής Δημοκρατίας"
    teychos_el = ET.SubElement(metadata_el, "Teychos")
    teychos_el.text = issue
    arithmos_el = ET.SubElement(metadata_el, "ArithmosFyllou")
    arithmos_el.text = issue_num
    day_el = ET.SubElement(metadata_el, "Hmerominia")
    day_el.text = pub_date

    periexomena = ET.SubElement(root, "Periexomena")
    kefalaio = ET.SubElement(periexomena, "Kefalaio")
    if re.search(r"ΠΡΑΞΕΙΣ\s+ΥΠΟΥΡΓΙΚΟΥ\s+ΣΥΜΒΟΥΛΙΟΥ", text, flags=re.I):
        kefalaio.set("titlos", "ΠΡΑΞΕΙΣ ΥΠΟΥΡΓΙΚΟΥ ΣΥΜΒΟΥΛΙΟΥ")
    else:
        kefalaio.set("titlos", data.get("document_type", "fek"))

    praxi = ET.SubElement(kefalaio, "Praxi")
    if action_number:
        praxi.set("arithmos", action_number)
    if action_date:
        praxi.set("hmerominia", action_date)

    praxi_title = ET.SubElement(praxi, "Titlos")
    praxi_title.text = action_title or data.get("title", "")

    if organ:
        organ_el = ET.SubElement(praxi, "Organ")
        organ_el.text = organ

    if aitiologia:
        aitiologia_el = ET.SubElement(praxi, "AitiolologikoMeros")
        eisagogi_el = ET.SubElement(aitiologia_el, "Eisagogi")
        eisagogi_el.text = "Έχοντας υπόψη:"
        for item in _split_numbered_items(aitiologia):
            stoicheio_el = ET.SubElement(aitiologia_el, "Stoicheio")
            stoicheio_el.set("arithmos", item["arithmos"])
            text_value = item["text"]
            m = re.match(r"(.+?:)\s*(.*)", text_value, flags=re.S)
            if m:
                eisagogi_item = ET.SubElement(stoicheio_el, "Eisagogi")
                eisagogi_item.text = m.group(1).strip()
                remainder = m.group(2).strip()
            else:
                remainder = text_value
            if list_items := _split_alpha_list(remainder):
                lista_el = ET.SubElement(stoicheio_el, "Lista")
                for entry in list_items:
                    peripton_el = ET.SubElement(lista_el, "Peripton")
                    peripton_el.set("grammato", entry["grammato"])
                    peripton_el.text = entry["text"]
            else:
                stoicheio_el.text = remainder

    if decision_text:
        apofasi_el = ET.SubElement(praxi, "ApoφασιστικόΜέρος")
        eisagogi_dec = ET.SubElement(apofasi_el, "Eisagogi")
        eisagogi_dec.text = "αποφασίζει:"
        if "Η παρούσα Πράξη να δημοσιευθεί" in decision_text:
            parts = decision_text.split("Η παρούσα Πράξη να δημοσιευθεί", 1)
            arthro_el = ET.SubElement(apofasi_el, "Arthro")
            arthro_el.text = parts[0].strip()
            diatagh_el = ET.SubElement(apofasi_el, "DiataghDimosiefseos")
            diatagh_el.text = "Η παρούσα Πράξη να δημοσιευθεί στην Εφημερίδα της Κυβέρνησης."
        else:
            arthro_el = ET.SubElement(apofasi_el, "Arthro")
            arthro_el.text = decision_text.strip()

    if signature_info.get("prothypourgos") or signature_info.get("members"):
        ypografes_el = ET.SubElement(praxi, "Ypografes")
        if signature_info.get("prothypourgos"):
            pro_el = ET.SubElement(ypografes_el, "Prothypourgos")
            onoma_el = ET.SubElement(pro_el, "Onoma")
            onoma_el.text = signature_info["prothypourgos"]
        if signature_info.get("members"):
            members_el = ET.SubElement(ypografes_el, "MeliYpourgikoySymvouliou")
            for member in signature_info["members"]:
                melos_el = ET.SubElement(members_el, "Melos")
                melos_el.text = member

    if contact_info["address"] or contact_info["phone"] or contact_info["url"]:
        epikoinonia_el = ET.SubElement(root, "Epikoinonia")
        if contact_info["address"]:
            dieuth_el = ET.SubElement(epikoinonia_el, "Dieuthinsi")
            dieuth_el.text = contact_info["address"]
        if contact_info["phone"]:
            tilel_el = ET.SubElement(epikoinonia_el, "Tilefono")
            tilel_el.text = contact_info["phone"]
        if contact_info["url"]:
            url_el = ET.SubElement(epikoinonia_el, "HlektronikaDieuthinsi")
            url_el.text = contact_info["url"]

    return ET.tostring(
        root, pretty_print=True, xml_declaration=True, encoding="utf-8"
    ).decode("utf-8")


def _should_build_efimerida(data: Dict[str, Any], raw_text: Optional[str]) -> bool:
    if data.get("document_type") == "cabinet_act":
        return True

    title = str(data.get("title", ""))
    if re.search(r"Πράξη|ΠΡΑΞΗ|Υπουργικού\s+Συμβουλίου|Υπουργικό\s+Συμβούλιο", title, flags=re.I):
        return True

    if raw_text:
        if re.search(r"ΕΦΗΜΕΡΙΔΑ(ΤΗΣ|\s+ΤΗΣ)?\s+ΚΥΒΕΡΝΗΣΕΩΣ|ΠΡΑΞΕΙΣ\s+ΥΠΟΥΡΓΙΚΟΥ\s+ΣΥΜΒΟΥΛΙΟΥ|Πράξη\s+Υπουργικού\s+Συμβουλίου|Υπουργικού\s+Συμβουλίου", raw_text, flags=re.I):
            return True

    for sec in data.get("sections", []):
        text = str(sec.get("text", ""))
        if re.search(r"Ο\s*Πρωθυπουργός|Τα\s*Μέλη\s*του\s*Υπουργικού\s*Συμβουλίου|Πράξη\s+Υπουργικού\s+Συμβουλίου", text, flags=re.I):
            return True

    return False


def to_xml(data: Dict[str, Any], raw_text: Optional[str] = None) -> str:
    # Priority 1: Use new cabinet_act structure if AI provided it
    if data.get("document_structure") == "cabinet_act" and data.get("action_info"):
        try:
            return _build_fek_cabinet_act_xml(data)
        except Exception as e:
            # Log the error but fallback gracefully
            import sys
            print(f"DEBUG: FEK cabinet_act builder failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
    
    # Priority 2: Try old regex-based builder for FEK documents
    if _should_build_efimerida(data, raw_text):
        try:
            return _build_efimerida_xml(data, raw_text or "")
        except Exception:
            pass

    # Fallback: Generic XML structure
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
    lines = ["<!doctype html>", "<html><head><meta charset='utf-8'>"]
    lines.append("<style>.diff-added { color: #00703c; background-color: #d7f1e6; } .diff-removed { color: #d4351c; background-color: #f9d6d2; } .fek-diff { font-family: monospace; white-space: pre-wrap; background-color: #f3f2f1; padding: 10px; border: 1px solid #b1b4b6; }</style>")
    lines.append("</head><body style='font-family: sans-serif;'>")
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

    if data.get("amendments"):
        lines.append("<h2>Amendments (Diff View)</h2>")
        for amend in data["amendments"]:
            if amend.get("diff"):
                lines.append(f"<h3>Change to {amend.get('target_law_id')}</h3>")
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
    data = structure_raw_text(raw_text, log_hook=log_hook)

    review_questions: List[str] = []
    for amend in data.get("amendments", []):
        if (amend.get("confidence") or 0) < 0.75 and amend.get("question_for_human"):
            review_questions.append(amend["question_for_human"])

    if data.get("metadata", {}).get("parse_error"):
        review_questions.append(
            "AI returned unparseable output – full manual review required."
        )

    needs_review = len(review_questions) > 0
    return StructuredOutput(
        data=data,
        xml_text=to_xml(data, raw_text),
        html_text=to_html(data),
        needs_review=needs_review,
        review_questions=review_questions,
    )
