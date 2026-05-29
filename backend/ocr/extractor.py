"""
OCR Extractor - Εξάγει κείμενο από σελίδες με text layer (PyMuPDF).
Χειρίζεται επίσης τις δύο στήλες: τις χωρίζει και τις ενώνει σωστά.
"""
import fitz
from typing import List, Optional
from .detector import PageInfo


def extract_text_layer(page: fitz.Page, info: PageInfo) -> str:
    """
    Εξάγει κείμενο από σελίδα που έχει text layer.
    Αν έχει δύο στήλες, τις χειρίζεται χωριστά.
    """
    if info.has_two_columns and info.column_x:
        return _extract_two_columns(page, info.column_x)
    else:
        return _extract_single_column(page)


def _extract_single_column(page: fitz.Page) -> str:
    """
    Απλή εξαγωγή κειμένου - διατηρεί σειρά ανάγνωσης.
    """
    # "dict" mode δίνει blocks με θέση για σωστή σειρά
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    lines_by_y = []

    for block in blocks:
        if block["type"] != 0:  # 0 = text block
            continue
        for line in block["lines"]:
            y = line["bbox"][1]  # top-y του line
            text = "".join(span["text"] for span in line["spans"])
            lines_by_y.append((y, text))

    # Ταξινόμηση κατά y (από πάνω προς τα κάτω)
    lines_by_y.sort(key=lambda x: x[0])
    return "\n".join(t for _, t in lines_by_y).strip()


def _extract_two_columns(page: fitz.Page, column_x: int) -> str:
    """
    Χωρίζει τη σελίδα στο column_x και εξάγει αριστερά + δεξιά χωριστά,
    μετά τα ενώνει: πρώτα αριστερά, μετά δεξιά.
    """
    rect = page.rect
    left_rect  = fitz.Rect(rect.x0, rect.y0, column_x, rect.y1)
    right_rect = fitz.Rect(column_x, rect.y0, rect.x1, rect.y1)

    left_text  = page.get_text("text", clip=left_rect).strip()
    right_text = page.get_text("text", clip=right_rect).strip()

    parts = []
    if left_text:
        parts.append(left_text)
    if right_text:
        parts.append(right_text)

    # ΦΕΚ: οι στήλες είναι συνεχής ροή ανάγνωσης (αριστερά → δεξιά).
    # Ενώνουμε με απλό newline — χωρίς ορατό marker που μολύνει το πιστό κείμενο.
    return "\n".join(parts)


def extract_document_text_layer(pdf_path: str, page_infos: List[PageInfo]) -> List[dict]:
    """
    Εξάγει κείμενο από ΟΛΕς τις σελίδες που έχουν text layer.
    Επιστρέφει λίστα από dicts: {page_num, text, method}
    """
    doc = fitz.open(pdf_path)
    results = []

    for info in page_infos:
        if info.is_scanned:
            # Θα το χειριστεί το engine.py (Tesseract)
            results.append({
                "page_num": info.page_num,
                "text": None,
                "method": "tesseract_pending"
            })
            continue

        page = doc[info.page_num - 1]
        text = extract_text_layer(page, info)
        results.append({
            "page_num": info.page_num,
            "text": text,
            "method": "pymupdf"
        })

    doc.close()
    return results