"""
OCR Extractor - Εξάγει κείμενο από σελίδες με text layer (PyMuPDF).
Χειρίζεται δύο στήλες, embedded images και tables.

Σειρά επεξεργασίας ανά σελίδα:
  1. Εντοπισμός image blocks → αποθήκευση PNG + placeholder [ΕΙΚΟΝΑ: ...]
  2. Εντοπισμός tables (find_tables) → Markdown + placeholder [ΠΙΝΑΚΑΣ]
  3. Υπόλοιπα text blocks → column-aware εξαγωγή
"""
import os
import fitz
from typing import List, Optional, Tuple
from .detector import PageInfo

try:
    from backend.config import IMAGES_DIR
except ImportError:
    try:
        from config import IMAGES_DIR
    except ImportError:
        IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "images")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _rect_overlap(r1: fitz.Rect, r2: fitz.Rect, threshold: float = 0.5) -> bool:
    """True αν τα δύο rects αλληλεπικαλύπτονται πάνω από threshold του μικρότερου."""
    inter = r1 & r2
    if inter.is_empty:
        return False
    area_inter = inter.width * inter.height
    area_min = min(r1.width * r1.height, r2.width * r2.height)
    return area_min > 0 and (area_inter / area_min) >= threshold


def _table_to_markdown(table) -> str:
    """Μετατρέπει PyMuPDF Table object σε Markdown."""
    try:
        rows = table.extract()
    except Exception:
        return "[ΠΙΝΑΚΑΣ: αδύνατη εξαγωγή]"

    if not rows:
        return ""

    lines = []
    for i, row in enumerate(rows):
        cells = [str(cell or "").replace("\n", " ").strip() for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("| " + " | ".join("---" for _ in cells) + " |")

    return "\n".join(lines)


# ── Κύριο extraction ──────────────────────────────────────────────────────────

def extract_text_layer(
    page: fitz.Page,
    info: PageInfo,
    image_out_dir: Optional[str] = None,
    page_label: str = "",
) -> str:
    """
    Εξάγει κείμενο από σελίδα που έχει text layer.
    - Embedded images → PNG αρχείο + [ΕΙΚΟΝΑ: filename] placeholder
    - Tables (find_tables) → Markdown
    - Text blocks → column-aware, σωστή σειρά ανάγνωσης
    """
    out_dir = image_out_dir or IMAGES_DIR
    os.makedirs(out_dir, exist_ok=True)

    # 1. Εντοπισμός και αποθήκευση images ────────────────────────────────────
    image_placeholders: List[Tuple[fitz.Rect, str]] = []  # (rect, placeholder_text)

    doc = page.parent
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        rects = page.get_image_rects(xref)
        for img_rect in rects:
            # Αποθήκευση: clip το pixmap του image
            try:
                clip_pix = page.get_pixmap(clip=img_rect, dpi=150)
                fname = f"{page_label}_img{xref}.png" if page_label else f"img{xref}.png"
                fpath = os.path.join(out_dir, fname)
                clip_pix.save(fpath)
                placeholder = f"[ΕΙΚΟΝΑ: {fname}]"
            except Exception:
                placeholder = f"[ΕΙΚΟΝΑ: xref={xref}]"
            image_placeholders.append((fitz.Rect(img_rect), placeholder))

    # 2. Εντοπισμός tables ────────────────────────────────────────────────────
    # Φίλτρο: τουλάχιστον 2 στήλες ΚΑΙ 2 γραμμές για να αποφύγουμε
    # false-positives από double-column text ή πίνακες περιεχομένων.
    table_placeholders: List[Tuple[fitz.Rect, str]] = []

    try:
        tabs = page.find_tables()
        for table in tabs.tables:
            if table.col_count < 2 or table.row_count < 2:
                continue
            md = _table_to_markdown(table)
            if md:
                table_placeholders.append((fitz.Rect(table.bbox), md))
    except Exception:
        pass  # find_tables μπορεί να μην υπάρχει σε παλαιότερες εκδόσεις PyMuPDF

    # 3. Εξαγωγή text blocks — παράλειψη αν ανήκουν σε image/table region ──
    excluded_rects = [r for r, _ in image_placeholders] + [r for r, _ in table_placeholders]

    if info.has_two_columns and info.column_x:
        text_part = _extract_two_columns(page, info.column_x, excluded_rects)
    else:
        text_part = _extract_single_column(page, excluded_rects)

    # 4. Ένωση: τοποθετούμε placeholders ΠΡΙΝ το υπόλοιπο κείμενο της σελίδας
    #    (απλοποιημένο: δεν κάνουμε in-flow injection — τα βάζουμε ομαδοποιημένα στην αρχή)
    prefix_parts = []
    for _, ph in image_placeholders:
        prefix_parts.append(ph)
    for _, ph in table_placeholders:
        prefix_parts.append(ph)

    parts = prefix_parts + ([text_part] if text_part.strip() else [])
    return "\n".join(parts)


def _extract_single_column(page: fitz.Page, excluded_rects: List[fitz.Rect] = None) -> str:
    """
    Απλή εξαγωγή κειμένου — διατηρεί σειρά ανάγνωσης.
    Παραλείπει blocks που ανήκουν σε excluded_rects (image/table περιοχές).
    """
    excluded = excluded_rects or []
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    lines_by_y = []

    for block in blocks:
        if block["type"] != 0:  # 0 = text block
            continue
        block_rect = fitz.Rect(block["bbox"])
        if any(_rect_overlap(block_rect, ex) for ex in excluded):
            continue
        for line in block["lines"]:
            y = line["bbox"][1]
            text = "".join(span["text"] for span in line["spans"])
            lines_by_y.append((y, text))

    lines_by_y.sort(key=lambda x: x[0])
    return "\n".join(t for _, t in lines_by_y).strip()


def _extract_two_columns(
    page: fitz.Page,
    column_x: int,
    excluded_rects: List[fitz.Rect] = None,
) -> str:
    """
    Χωρίζει τη σελίδα στο column_x και εξάγει αριστερά + δεξιά χωριστά.
    Παραλείπει regions εικόνων/tables (clip δεν τις εξαιρεί, οπότε χρησιμοποιούμε blocks).
    """
    excluded = excluded_rects or []
    rect = page.rect

    def _col_text(x0: float, x1: float) -> str:
        clip = fitz.Rect(x0, rect.y0, x1, rect.y1)
        blocks = page.get_text("dict", clip=clip, flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        lines_by_y = []
        for block in blocks:
            if block["type"] != 0:
                continue
            block_rect = fitz.Rect(block["bbox"])
            if any(_rect_overlap(block_rect, ex) for ex in excluded):
                continue
            for line in block["lines"]:
                y = line["bbox"][1]
                text = "".join(span["text"] for span in line["spans"])
                lines_by_y.append((y, text))
        lines_by_y.sort(key=lambda x: x[0])
        return "\n".join(t for _, t in lines_by_y).strip()

    left_text = _col_text(rect.x0, column_x)
    right_text = _col_text(column_x, rect.x1)

    parts = [p for p in [left_text, right_text] if p]
    return "\n".join(parts)


def extract_document_text_layer(pdf_path: str, page_infos: List[PageInfo]) -> List[dict]:
    """
    Εξάγει κείμενο από ΟΛΕς τις σελίδες που έχουν text layer.
    Επιστρέφει λίστα από dicts: {page_num, text, method}
    """
    doc = fitz.open(pdf_path)
    # Βάση ονόματος για τα images (από το pdf filename)
    pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]
    results = []

    for info in page_infos:
        if info.is_scanned:
            results.append({
                "page_num": info.page_num,
                "text": None,
                "method": "tesseract_pending"
            })
            continue

        page = doc[info.page_num - 1]
        page_label = f"{pdf_base}_p{info.page_num}"
        text = extract_text_layer(page, info, page_label=page_label)
        results.append({
            "page_num": info.page_num,
            "text": text,
            "method": "pymupdf"
        })

    doc.close()
    return results
