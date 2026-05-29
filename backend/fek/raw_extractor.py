"""
Raw Extractor — πιστή εξαγωγή ΦΕΚ (faithful extraction, NO AI).

Συνδέει τα ΥΠΑΡΧΟΝΤΑ εργαλεία:
  - detector.detect_document         → geometry-based column / scanned detection
  - extractor.extract_document_text_layer → column-aware raw text σε σωστή σειρά
  - engine.ocr_document              → Tesseract OCR για scanned σελίδες

Το κείμενο διατηρείται αυτούσιο (verbatim). Κανένα summarization/reformat.
Τα metadata εξάγονται με regex από το masthead (NO AI).
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional

from lxml import etree as ET

# ── Import-fallback pattern (όπως στα υπόλοιπα modules) ─────────────────────────
try:
    from backend.ocr.detector import detect_document
    from backend.ocr.extractor import extract_document_text_layer
    from backend.ocr.engine import ocr_document
except Exception:  # pragma: no cover - allow alternate package layout
    try:
        from ocr.detector import detect_document
        from ocr.extractor import extract_document_text_layer
        from ocr.engine import ocr_document
    except Exception:
        detect_document = None  # type: ignore
        extract_document_text_layer = None  # type: ignore
        ocr_document = None  # type: ignore

# Επαναχρησιμοποίηση των date helpers που ήδη υπάρχουν στον structurer.
try:
    from backend.ai.structurer import greek_months, _format_greek_date, _format_dot_date
except Exception:  # pragma: no cover
    try:
        from ai.structurer import greek_months, _format_greek_date, _format_dot_date
    except Exception:
        # Μικρό local fallback ώστε το module να εισάγεται/δουλεύει χωρίς deps.
        greek_months = {
            "Ιανουαρίου": 1, "Φεβρουαρίου": 2, "Μαρτίου": 3, "Απριλίου": 4,
            "Μαΐου": 5, "Ιουνίου": 6, "Ιουλίου": 7, "Αυγούστου": 8,
            "Σεπτεμβρίου": 9, "Οκτωβρίου": 10, "Νοεμβρίου": 11, "Δεκεμβρίου": 12,
        }

        def _format_greek_date(match: str) -> Optional[str]:
            match = match.strip()
            if m := re.search(r"(\d{1,2})\s+([Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώ]+)\s+(\d{4})", match):
                month = greek_months.get(m.group(2))
                if month:
                    return f"{int(m.group(3)):04d}-{month:02d}-{int(m.group(1)):02d}"
            return None

        def _format_dot_date(match: str) -> Optional[str]:
            match = match.strip()
            if m := re.search(r"(\d{2})\.(\d{2})\.(\d{4})", match):
                return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
            return None


# ── 1. Faithful raw extraction ──────────────────────────────────────────────────

def extract_raw_text(pdf_path: str, log_hook: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
    """
    Πιστή εξαγωγή RAW κειμένου με τα ΥΠΑΡΧΟΝΤΑ detector + extractor + OCR engine.

    - detect_document(pdf_path) → page_infos
    - extract_document_text_layer → ανά-σελίδα αποτελέσματα (text layer)
    - Για σελίδες με method=="tesseract_pending" (scanned) → Tesseract OCR engine.
    - Ένωση σελίδων στη σειρά. Κείμενο αυτούσιο. Κανένα summary/reformat.

    Returns: {"raw_text": str, "pages": [{page_num, text, method}], "page_count": int}
    """
    def _log(msg: str) -> None:
        if log_hook:
            log_hook("extract", msg)

    if detect_document is None or extract_document_text_layer is None:
        raise RuntimeError(
            "Missing runtime dependency (PyMuPDF/OpenCV); install project dependencies."
        )

    page_infos = detect_document(pdf_path)
    _log(f"detected {len(page_infos)} page(s)")

    results = extract_document_text_layer(pdf_path, page_infos)

    # Scanned σελίδες: συμπλήρωση με OCR (Tesseract) από το engine.
    pending = [r for r in results if r.get("method") == "tesseract_pending"]
    if pending:
        _log(f"running Tesseract OCR on {len(pending)} scanned page(s)")
        ocr_results = ocr_document(pdf_path, page_infos) if ocr_document else []
        ocr_by_page = {r["page_num"]: r for r in ocr_results}
        for r in results:
            if r.get("method") == "tesseract_pending":
                ocr = ocr_by_page.get(r["page_num"])
                if ocr is not None:
                    r["text"] = ocr.get("text") or ""
                    r["method"] = ocr.get("method", "tesseract")
                else:
                    r["text"] = ""
                    r["method"] = "tesseract"

    # Σειρά ανά page_num, ένωση αυτούσια.
    pages = sorted(results, key=lambda r: r["page_num"])
    raw_text = "\n\n".join((p.get("text") or "") for p in pages)

    return {
        "raw_text": raw_text,
        "pages": pages,
        "page_count": len(pages),
    }


# ── 2. Regex-based metadata extraction (NO AI) ──────────────────────────────────

_TEYCHOS_MAP = {
    "ΠΡΩΤΟ": "Α'",
    "ΔΕΥΤΕΡΟ": "Β'",
    "ΤΡΙΤΟ": "Γ'",
    "ΤΕΤΑΡΤΟ": "Δ'",
}


def _normalize_teychos(value: str) -> Optional[str]:
    """Κανονικοποίηση τεύχους σε "Α'"/"Β'"/"Γ'"/"Δ'"."""
    v = value.strip().upper()
    # Word form: ΠΡΩΤΟ → Α' κ.λπ.
    if v in _TEYCHOS_MAP:
        return _TEYCHOS_MAP[v]
    # Letter form: Α / Α' / Α’ → Α'
    letter = v.rstrip("'’").strip()
    if letter in ("Α", "Β", "Γ", "Δ"):
        return f"{letter}'"
    return None


def extract_metadata(raw_text: str) -> Dict[str, Any]:
    """
    Regex-based metadata extraction από το masthead του ΦΕΚ. NO AI.

    Returns dict (None/empty όταν δεν βρεθεί):
      titlos, teychos, arithmos_fyllou, hmerominia (YYYY-MM-DD), document_type
    """
    # Συμπύκνωση whitespace για σταθερά masthead matches (το raw_text μένει αυτούσιο αλλού).
    flat = re.sub(r"\s+", " ", raw_text.replace("\r", " ").replace("\n", " ")).strip()

    metadata: Dict[str, Any] = {
        "titlos": None,
        "teychos": None,
        "arithmos_fyllou": None,
        "hmerominia": None,
        "document_type": "unknown",
    }

    # Titlos — μόνο αν ταιριάζει το masthead.
    if re.search(r"ΕΦΗΜΕΡΙΔΑ\s+ΤΗΣ\s+ΚΥΒΕΡΝΗΣΕΩΣ", flat, flags=re.I):
        metadata["titlos"] = "Εφημερίδα της Κυβερνήσεως της Ελληνικής Δημοκρατίας"

    # Teychos — "ΤΕΥΧΟΣ ΠΡΩΤΟ" / "ΤΕΥΧΟΣ Α'" / "Τεύχος Α'".
    if m := re.search(r"ΤΕΥΧΟΣ\s+([Α-Ωα-ω]+['’]?)", flat, flags=re.I):
        metadata["teychos"] = _normalize_teychos(m.group(1))

    # ArithmosFyllou — ψηφία μετά το "Αρ. Φύλλου".
    if m := re.search(r"Αρ\.?\s*Φύλλου\s*(\d+)", flat, flags=re.I):
        metadata["arithmos_fyllou"] = m.group(1)

    # Hmerominia — ελληνική ημερομηνία ή dot-date → YYYY-MM-DD.
    if m := re.search(r"(\d{1,2}\s+[Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώ]+\s+\d{4})", flat):
        metadata["hmerominia"] = _format_greek_date(m.group(1))
    elif m := re.search(r"(\d{2}\.\d{2}\.\d{4})", flat):
        metadata["hmerominia"] = _format_dot_date(m.group(1))

    # Document type — best-effort.
    if re.search(r"ΠΡΑΞΕΙΣ\s+ΥΠΟΥΡΓΙΚΟΥ\s+ΣΥΜΒΟΥΛΙΟΥ|ΤΟ\s+ΥΠΟΥΡΓΙΚΟ\s+ΣΥΜΒΟΥΛΙΟ", flat, flags=re.I):
        metadata["document_type"] = "praxi_ypourgikou_symvouliou"
    elif re.search(r"ΝΟΜΟΣ\s+ΥΠ['’]?\s*ΑΡΙΘ", flat, flags=re.I):
        metadata["document_type"] = "nomos"
    elif re.search(r"ΠΡΟΕΔΡΙΚΟ\s+ΔΙΑΤΑΓΜΑ", flat, flags=re.I):
        metadata["document_type"] = "proedriko_diatagma"

    return metadata


# ── 3. Light archival XML ───────────────────────────────────────────────────────

# XML 1.0 επιτρέπει μόνο: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
_XML_ILLEGAL = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff￾￿]"
)


def _sanitize_for_xml(text: str) -> str:
    """Αφαιρεί χαρακτήρες που δεν επιτρέπονται σε XML 1.0 (π.χ. NULL, control chars από OCR)."""
    if not isinstance(text, str):
        return ""
    return _XML_ILLEGAL.sub("�", text)


def _sx(metadata: Dict[str, Any], key: str) -> str:
    """Εξάγει και sanitarίζει ένα metadata string για χρήση σε XML."""
    return _sanitize_for_xml(metadata.get(key) or "")


def build_archive_xml(metadata: Dict[str, Any], raw_text: str) -> str:
    """
    Φτιάχνει ΕΛΑΦΡΥ, πιστό αρχειακό XML (ΟΧΙ το παλιό βαρύ structured schema).

    <FEK>
      <Metadata>
        <Titlos/> <Teychos/> <ArithmosFyllou/> <Hmerominia/> <Typos/>
      </Metadata>
      <Keimeno> raw text αυτούσιο </Keimeno>
    </FEK>
    """
    root = ET.Element("FEK")

    metadata_el = ET.SubElement(root, "Metadata")
    ET.SubElement(metadata_el, "Titlos").text        = _sx(metadata, "titlos")
    ET.SubElement(metadata_el, "Teychos").text       = _sx(metadata, "teychos")
    ET.SubElement(metadata_el, "ArithmosFyllou").text = _sx(metadata, "arithmos_fyllou")
    ET.SubElement(metadata_el, "Hmerominia").text    = _sx(metadata, "hmerominia")
    ET.SubElement(metadata_el, "Typos").text         = _sx(metadata, "document_type")

    ET.SubElement(root, "Keimeno").text = _sanitize_for_xml(raw_text)

    return ET.tostring(
        root, pretty_print=True, xml_declaration=True, encoding="utf-8"
    ).decode("utf-8")


# ── 4. Simple readable HTML ─────────────────────────────────────────────────────

def _esc(v: Any) -> str:
    return (
        str(v)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _raw_text_to_html_body(raw_text: str, images_url_prefix: str = "images") -> str:
    """
    Μετατρέπει raw text σε HTML body content:
    - [ΕΙΚΟΝΑ: fname.png] → <img src="images/fname.png">
    - Markdown tables (| ... |) → <table>
    - Υπόλοιπο κείμενο → <pre>
    """
    import re

    # Split σε "chunks" ανά placeholder/table/text
    chunks = []
    remaining = raw_text

    # Regex: image placeholder ή markdown table block ή plain text
    pattern = re.compile(
        r"(\[ΕΙΚΟΝΑ:\s*([^\]]+)\])"           # group 1/2: image
        r"|(\n?\|.+\|(?:\n\|.+\|)*)",         # group 3: markdown table
        re.DOTALL,
    )

    last = 0
    for m in pattern.finditer(remaining):
        # text πριν το match
        before = remaining[last:m.start()]
        if before.strip():
            chunks.append(("text", before))

        if m.group(1):  # ΕΙΚΟΝΑ
            fname = m.group(2).strip()
            chunks.append(("image", fname))
        elif m.group(3):  # table
            chunks.append(("table", m.group(3).strip()))

        last = m.end()

    tail = remaining[last:]
    if tail.strip():
        chunks.append(("text", tail))

    html_parts = []
    for kind, content in chunks:
        if kind == "text":
            html_parts.append(f'<pre class="keimeno">{_esc(content)}</pre>')
        elif kind == "image":
            src = f"{images_url_prefix}/{_esc(content)}"
            alt = _esc(content)
            html_parts.append(
                f'<figure class="fek-image">'
                f'<img src="{src}" alt="{alt}" style="max-width:100%;">'
                f'<figcaption>{alt}</figcaption></figure>'
            )
        elif kind == "table":
            html_parts.append(_markdown_table_to_html(content))

    return "\n".join(html_parts)


def _markdown_table_to_html(md: str) -> str:
    """Μετατρέπει Markdown table σε HTML <table>."""
    import re
    rows = [line.strip() for line in md.strip().splitlines() if line.strip()]
    html = ['<table class="fek-table" border="1" cellpadding="4" cellspacing="0">']
    for i, row in enumerate(rows):
        if re.match(r"^\|[\s\-|]+\|$", row):
            continue  # separator row
        cells = [c.strip() for c in row.strip("|").split("|")]
        tag = "th" if i == 0 else "td"
        html.append("  <tr>" + "".join(f"<{tag}>{_esc(c)}</{tag}>" for c in cells) + "</tr>")
    html.append("</table>")
    return "\n".join(html)


def build_archive_html(metadata: Dict[str, Any], raw_text: str, images_url_prefix: str = "images") -> str:
    """
    Απλό, αναγνώσιμο HTML: header με metadata + raw κείμενο.
    - [ΕΙΚΟΝΑ: fname] → <img>
    - Markdown tables → <table>
    - Υπόλοιπο → <pre> (διατηρεί formatting). Ελληνική serif τυπογραφία.
    """
    titlos = metadata.get("titlos") or "ΦΕΚ"
    rows = [
        ("Τεύχος", metadata.get("teychos")),
        ("Αρ. Φύλλου", metadata.get("arithmos_fyllou")),
        ("Ημερομηνία", metadata.get("hmerominia")),
        ("Τύπος", metadata.get("document_type")),
    ]
    meta_rows = "\n".join(
        f"      <tr><th style='text-align:left;padding-right:1em;'>{_esc(label)}</th>"
        f"<td>{_esc(value or '')}</td></tr>"
        for label, value in rows
    )

    body_content = _raw_text_to_html_body(raw_text, images_url_prefix)

    return f"""<!doctype html>
<html lang="el">
<head>
<meta charset="utf-8">
<title>{_esc(titlos)}</title>
<style>
  body {{ font-family: 'GFS Didot', 'Times New Roman', Georgia, serif;
          color: #1a1a1a; background: #fff; margin: 2em auto; max-width: 50em;
          line-height: 1.6; }}
  header {{ border-bottom: 2px solid #1a1a1a; margin-bottom: 1.5em; padding-bottom: 1em; }}
  h1 {{ font-size: 1.4em; margin: 0 0 0.5em; }}
  table.meta {{ font-size: 0.95em; }}
  pre.keimeno {{ white-space: pre-wrap; word-wrap: break-word;
                 font-family: 'GFS Didot', 'Times New Roman', Georgia, serif;
                 font-size: 1.05em; }}
  table.fek-table {{ border-collapse: collapse; width: 100%; margin: 1em 0;
                     font-size: 0.95em; }}
  figure.fek-image {{ text-align: center; margin: 1.5em 0; }}
  figure.fek-image figcaption {{ font-size: 0.8em; color: #666; margin-top: 0.3em; }}
</style>
</head>
<body>
  <header>
    <h1>{_esc(titlos)}</h1>
    <table class="meta">
{meta_rows}
    </table>
  </header>
  <main>
{body_content}
  </main>
</body>
</html>"""
