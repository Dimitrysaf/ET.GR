"""Regex-based parser for Greek statutory amendment instructions.

ΦΕΚ amendment sentences are highly formulaic but rarely quote the OLD text.
They give a LOCATOR (άρθρο/παρ./περ./εδάφιο + target law id) plus the NEW
text in Greek quotes «...». This module extracts those instructions with
pure regex — NO AI. A dormant Claude-based hook is provided for ambiguous
text but returns None cleanly when no API key is configured.

Παραδείγματα / Examples:
  - «Η παρ. 3 του άρθρου 46 του ν. 4622/2019 αντικαθίσταται ως εξής: «...».»
  - «Στο άρθρο 12 του ν. 4622/2019 προστίθεται παράγραφος 5 ως εξής: «...».»
  - «Η παρ. 2 του άρθρου 5 του ν. 4622/2019 καταργείται.»
  - «Η περ. β' της παρ. 1 του άρθρου 9 του ν. 4622/2019 διαγράφεται.»
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ParsedAmendment:
    target_law_id: str          # "ν. 4622/2019"
    operation: str              # "replace" | "insert" | "delete"
    article: Optional[str]      # "46"
    paragraph: Optional[str]    # "3"
    periptosi: Optional[str]    # "β" (περ.) or None
    edafio: bool                # True if "εδάφιο" mentioned
    new_text: Optional[str]     # content from «...», or None
    old_text: Optional[str]     # usually None for Greek amendments
    confidence: float           # heuristic 0..1
    raw_instruction: str        # the matched sentence span


# ── Operation keywords → canonical operation ────────────────────────────────
# αντικαθίσταται → replace, προστίθεται → insert, καταργείται/διαγράφεται → delete
_OP_KEYWORDS = [
    ("replace", re.compile(r"αντικαθίστα(?:ται|νται)", re.IGNORECASE)),
    ("insert", re.compile(r"προστίθε(?:ται|νται)", re.IGNORECASE)),
    ("delete", re.compile(r"(?:καταργείται|καταργούνται|διαγράφε(?:ται|νται))", re.IGNORECASE)),
]

# ── Target law id ───────────────────────────────────────────────────────────
# Δέχεται "ν. 4622/2019", "ν.δ. 356/1974", "π.δ. 63/2005", "νόμου 4622/2019".
# Normalize prefix to a canonical short form: ν. / ν.δ. / π.δ.
_LAW_PREFIX_RE = (
    r"(?:ν\.\s*δ\.|ν\.δ\.|π\.\s*δ\.|π\.δ\.|"
    r"ν[\.όόουυ]*\s*|νόμου\s*|νόμος\s*|π[\.ρ]*\s*δ[\.ιάτγμαος]*\s*)"
)
_LAW_RE = re.compile(
    r"(?P<prefix>ν\.\s*δ\.|ν\.δ\.|π\.\s*δ\.|π\.δ\.|νόμου|νόμος|ν\.|π\.δ)"
    r"\s*(?P<num>\d{1,5})\s*/\s*(?P<year>\d{4})",
    re.IGNORECASE,
)

# ── Locator parts ───────────────────────────────────────────────────────────
_ARTICLE_RE = re.compile(r"άρθρ(?:ο|ου)\s*(\d{1,4}[Α-ΩA-Z]?)", re.IGNORECASE)
_PARAGRAPH_RE = re.compile(r"παρ(?:\.|αγράφου|άγραφος|αγράφο)\s*(\d{1,3})", re.IGNORECASE)
_PERIPTOSI_RE = re.compile(r"περ(?:\.|ίπτωση|ίπτωσης)\s*([α-ωΑ-Ω])", re.IGNORECASE)
_EDAFIO_RE = re.compile(r"εδάφι[οα]", re.IGNORECASE)

# ── "ως εξής:" followed by quoted new text ──────────────────────────────────
# Προτιμάμε ελληνικά εισαγωγικά « », με fallback σε ευθεία " ".
_GREEK_QUOTE_RE = re.compile(r"«(.*?)»", re.DOTALL)
_STRAIGHT_QUOTE_RE = re.compile(r"\"(.*?)\"", re.DOTALL)


def _canonical_law_id(prefix: str, num: str, year: str) -> str:
    """Normalize a matched law reference to a canonical id string.

    Επιστρέφει π.χ. "ν. 4622/2019", "ν.δ. 356/1974", "π.δ. 63/2005".
    """
    p = prefix.lower().replace(" ", "")
    if p.startswith("ν.δ") or p.startswith("ν.δ."):
        canon = "ν.δ."
    elif p.startswith("π.δ") or p.startswith("π."):
        canon = "π.δ."
    else:
        # ν., νόμου, νόμος
        canon = "ν."
    return f"{canon} {num}/{year}"


# Συντομογραφίες που τελειώνουν σε τελεία και ΔΕΝ πρέπει να σπάνε την πρόταση.
# Abbreviations whose trailing dot must not terminate a candidate span.
_ABBREVIATIONS = (
    "ν", "π", "δ", "παρ", "περ", "αρθρ", "αρ", "εδ", "στοιχ",
    "κ", "πρβλ", "βλ", "ΦΕΚ", "τευχ",
)


def _is_abbrev_dot(text: str, dot_idx: int) -> bool:
    """True αν η τελεία στη θέση dot_idx ανήκει σε γνωστή συντομογραφία.

    True if the dot at dot_idx is part of a known abbreviation (e.g. "παρ.",
    "ν.", "π.δ.") rather than a sentence terminator.
    """
    # Μάζεψε τα γράμματα ακριβώς πριν την τελεία.
    j = dot_idx - 1
    word_chars: List[str] = []
    while j >= 0 and (text[j].isalpha() or text[j] == "."):
        if text[j] == ".":
            break
        word_chars.append(text[j])
        j -= 1
    word = "".join(reversed(word_chars))
    return word in _ABBREVIATIONS


def _split_candidates(raw_text: str) -> List[str]:
    """Split text into candidate instruction spans.

    Σπάμε σε προτάσεις σε τελεία/άνω τελεία/νέα γραμμή, αλλά ΔΕΝ σπάμε μέσα
    σε ελληνικά εισαγωγικά «...», ούτε σε τελείες συντομογραφιών (π.χ. "παρ.",
    "ν.", "π.δ."), ώστε να μη χάσουμε ή τεμαχίσουμε την οδηγία τροποποίησης.
    """
    spans: List[str] = []
    buf: List[str] = []
    depth = 0
    i = 0
    n = len(raw_text)
    while i < n:
        ch = raw_text[i]
        if ch == "«":
            depth += 1
        elif ch == "»":
            depth = max(0, depth - 1)
        buf.append(ch)
        if depth == 0 and ch in "\n·;":
            spans.append("".join(buf).strip())
            buf = []
        elif depth == 0 and ch == "»" and len(buf) >= 2 and buf[-2] == ".":
            # Κλείσιμο εισαγωγικών μετά από τελεία (".»") = τέλος πρότασης.
            # Η συνήθης μορφή ΦΕΚ βάζει την τελεία ΜΕΣΑ στα εισαγωγικά, οπότε
            # δεν θα τη βρίσκαμε ποτέ ως όριο διαφορετικά.
            spans.append("".join(buf).strip())
            buf = []
        elif depth == 0 and ch == ".":
            # Σπάμε σε τελεία μόνο αν ΔΕΝ είναι συντομογραφία.
            if not _is_abbrev_dot(raw_text, i):
                spans.append("".join(buf).strip())
                buf = []
        i += 1
    if buf:
        spans.append("".join(buf).strip())
    return [s for s in spans if s]


def _extract_new_text(span: str) -> Optional[str]:
    """Extract the «...» (ή "...") new text that follows "ως εξής:"."""
    # Προτιμάμε ό,τι ακολουθεί το "ως εξής:".
    idx = span.find("ως εξής")
    region = span[idx:] if idx >= 0 else span
    m = _GREEK_QUOTE_RE.search(region)
    if m:
        return m.group(1).strip()
    m = _STRAIGHT_QUOTE_RE.search(region)
    if m:
        return m.group(1).strip()
    return None


def _parse_span(span: str) -> Optional[ParsedAmendment]:
    """Parse a single candidate span into a ParsedAmendment, or None."""
    op = None
    for name, rgx in _OP_KEYWORDS:
        if rgx.search(span):
            op = name
            break
    if op is None:
        return None

    law_m = _LAW_RE.search(span)
    if law_m is None:
        # Χωρίς αναγνωρίσιμο νόμο-στόχο δεν μπορούμε να κωδικοποιήσουμε.
        return None
    target_law_id = _canonical_law_id(
        law_m.group("prefix"), law_m.group("num"), law_m.group("year")
    )

    art_m = _ARTICLE_RE.search(span)
    par_m = _PARAGRAPH_RE.search(span)
    per_m = _PERIPTOSI_RE.search(span)

    article = art_m.group(1) if art_m else None
    paragraph = par_m.group(1) if par_m else None
    periptosi = per_m.group(1) if per_m else None
    edafio = bool(_EDAFIO_RE.search(span))

    # Για replace/insert περιμένουμε νέο κείμενο· για delete όχι.
    new_text = None
    if op in ("replace", "insert"):
        new_text = _extract_new_text(span)

    # ── Heuristic confidence ────────────────────────────────────────────────
    confidence = 0.5
    if article:
        confidence += 0.2
    if paragraph or periptosi:
        confidence += 0.1
    if op == "delete":
        # Οι καταργήσεις/διαγραφές δεν χρειάζονται νέο κείμενο.
        confidence += 0.15
    elif new_text:
        confidence += 0.2
    else:
        # replace/insert χωρίς νέο κείμενο → ασαφές.
        confidence -= 0.2
    confidence = max(0.0, min(1.0, confidence))

    return ParsedAmendment(
        target_law_id=target_law_id,
        operation=op,
        article=article,
        paragraph=paragraph,
        periptosi=periptosi,
        edafio=edafio,
        new_text=new_text,
        old_text=None,  # σχεδόν ποτέ δεν παρατίθεται το παλιό κείμενο
        confidence=confidence,
        raw_instruction=span.strip(),
    )


def parse_amendments(raw_text: str) -> List[ParsedAmendment]:
    """Regex-based extraction of statutory amendment instructions.

    Splits text into candidate sentences/clauses (without breaking inside
    «...»), detects the operation keyword, extracts target law id + locator +
    «new text». NO AI. Sets confidence lower when locator or new_text is
    missing/ambiguous. Returns [] if none found.
    """
    if not raw_text:
        return []
    out: List[ParsedAmendment] = []
    for span in _split_candidates(raw_text):
        parsed = _parse_span(span)
        if parsed is not None:
            out.append(parsed)
    return out


def _build_claude_prompt(raw_text: str) -> str:
    """Build the instruction prompt for the (dormant) Claude parser."""
    return (
        "You are a Greek legal text parser. Extract every statutory amendment "
        "instruction from the following ΦΕΚ text. For each, return a JSON array "
        "of objects with keys: target_law_id (e.g. 'ν. 4622/2019'), operation "
        "('replace'|'insert'|'delete'), article, paragraph, periptosi, edafio "
        "(bool), new_text (content inside «...» after 'ως εξής:', or null), "
        "old_text (usually null), confidence (0..1). Return ONLY JSON.\n\n"
        f"TEXT:\n{raw_text}"
    )


def claude_parse_amendments(raw_text: str, log_hook=None) -> Optional[List[ParsedAmendment]]:
    """OPTIONAL Claude-based parser for ambiguous text (DORMANT hook).

    Reads ANTHROPIC_API_KEY from env. Lazily imports `anthropic` inside a
    try/except. If the key OR the package is missing, returns None immediately
    (graceful no-op) and logs that Claude is unavailable. `anthropic` is NOT a
    project dependency — when present and a key is set, this would call the
    Messages API. With no key in the current environment, this path is dormant
    and always returns None cleanly.
    """
    def _log(msg: str) -> None:
        if log_hook:
            try:
                log_hook("claude_parse_amendments", msg)
            except Exception:
                pass

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _log("Claude unavailable: ANTHROPIC_API_KEY not set — skipping.")
        return None

    try:
        import anthropic  # type: ignore  # lazy, NOT in requirements.txt
    except Exception:
        _log("Claude unavailable: `anthropic` package not installed — skipping.")
        return None

    # ── Dormant live path ───────────────────────────────────────────────────
    # When a key and the SDK are both present this calls the Messages API and
    # maps the JSON response into ParsedAmendment objects. Wrapped defensively
    # so any failure degrades to None rather than raising.
    try:
        import json as _json

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": _build_claude_prompt(raw_text)}],
        )
        # Συνένωση όλων των text blocks της απάντησης.
        text = "".join(
            getattr(block, "text", "") for block in getattr(resp, "content", [])
        )
        data = _json.loads(text)
        out: List[ParsedAmendment] = []
        for item in data:
            out.append(
                ParsedAmendment(
                    target_law_id=item.get("target_law_id", ""),
                    operation=(item.get("operation") or "").lower(),
                    article=item.get("article"),
                    paragraph=item.get("paragraph"),
                    periptosi=item.get("periptosi"),
                    edafio=bool(item.get("edafio")),
                    new_text=item.get("new_text"),
                    old_text=item.get("old_text"),
                    confidence=float(item.get("confidence") or 0.5),
                    raw_instruction=item.get("raw_instruction", ""),
                )
            )
        _log(f"Claude parsed {len(out)} amendment(s).")
        return out
    except Exception as exc:  # pragma: no cover - dormant, no key in env
        _log(f"Claude parse failed: {exc}")
        return None
