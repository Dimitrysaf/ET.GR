"""
OCR Detector - Αναλύει κάθε σελίδα και αποφασίζει:
  - Έχει text layer (selectable text) ή είναι scanned;
  - Είναι mixed (μερικές περιοχές scanned μέσα σε selectable);
  - Έχει δύο στήλες / διαχωριστικό;

Returns ένα PageInfo dataclass για κάθε σελίδα.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import fitz          # PyMuPDF
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from backend.config import (
        MIN_TEXT_CONFIDENCE, MIN_CHARS_PER_PAGE,
        COLUMN_WHITE_RATIO, COLUMN_SEARCH_BAND
    )
except ImportError:
    from config import (
        MIN_TEXT_CONFIDENCE, MIN_CHARS_PER_PAGE,
        COLUMN_WHITE_RATIO, COLUMN_SEARCH_BAND
    )


# ── Αποτέλεσμα ανάλυσης ──────────────────────────────────────────────────────

@dataclass
class PageInfo:
    page_num: int
    has_text_layer: bool          # PyMuPDF βρήκε αρκετό κείμενο
    is_scanned: bool              # χρειάζεται Tesseract
    is_mixed: bool                # ένα μέρος text, ένα scanned
    has_two_columns: bool         # δύο στήλες με διαχωριστικό
    column_x: Optional[int]       # pixel x όπου χωρίζουν οι στήλες (αν υπάρχουν)
    char_count: int               # πόσοι χαρακτήρες βρέθηκαν από PyMuPDF
    image_regions: List[Tuple]    # (x0, y0, x1, y1) περιοχές που είναι εικόνα
    confidence_hint: float        # 0-1 εκτίμηση ποιότητας text layer


# ── Κύρια συνάρτηση ─────────────────────────────────────────────────────────

def detect_page(page: fitz.Page, page_num: int) -> PageInfo:
    """
    Αναλύει μία σελίδα PDF και επιστρέφει PageInfo.
    """
    # 1. Προσπάθεια εξαγωγής κειμένου με PyMuPDF
    text = page.get_text("text").strip()
    char_count = len(text)

    # 2. Έλεγχος αν υπάρχουν embedded images
    image_list = page.get_images(full=True)
    image_regions = _get_image_rects(page, image_list)

    # 3. Εκτίμηση ποιότητας text layer
    confidence = _estimate_text_quality(text, char_count)

    has_text_layer = (
        char_count >= MIN_CHARS_PER_PAGE
        and confidence >= MIN_TEXT_CONFIDENCE
    )

    # 4. Αν υπάρχουν εικόνες ΚΑΙ κείμενο → mixed
    is_mixed = has_text_layer and len(image_regions) > 0

    # 5. Αν δεν υπάρχει text layer → scanned
    is_scanned = not has_text_layer

    # 6. Column detection (χρειάζεται rendering)
    has_two_columns = False
    column_x = None
    if CV2_AVAILABLE:
        has_two_columns, column_x = _detect_columns(page)

    return PageInfo(
        page_num=page_num,
        has_text_layer=has_text_layer,
        is_scanned=is_scanned,
        is_mixed=is_mixed,
        has_two_columns=has_two_columns,
        column_x=column_x,
        char_count=char_count,
        image_regions=image_regions,
        confidence_hint=confidence,
    )


def detect_document(pdf_path: str) -> List[PageInfo]:
    """
    Αναλύει ολόκληρο PDF και επιστρέφει λίστα PageInfo, μία ανά σελίδα.
    """
    doc = fitz.open(pdf_path)
    results = []
    for i, page in enumerate(doc):
        info = detect_page(page, page_num=i + 1)
        results.append(info)
    doc.close()
    return results


# ── Βοηθητικές ───────────────────────────────────────────────────────────────

def _estimate_text_quality(text: str, char_count: int) -> float:
    """
    Απλή ευρετική εκτίμηση ποιότητας text layer.
    Ελέγχει αναλογία αναγνωρίσιμων χαρακτήρων vs σκουπίδια.
    """
    if char_count == 0:
        return 0.0

    # Μέτρα "σκουπιδιών": χαρακτήρες που δεν ανήκουν σε ελληνικά/λατινικά/αριθμούς/κοινά σημεία
    import re
    valid = re.findall(r'[\u0370-\u03FF\u1F00-\u1FFFa-zA-Z0-9\s.,;:\-–()/«»"\'%€]', text)
    ratio = len(valid) / max(char_count, 1)
    return round(ratio, 3)


def _get_image_rects(page: fitz.Page, image_list: list) -> List[Tuple]:
    """
    Επιστρέφει τα bounding boxes των embedded images σε μία σελίδα.
    """
    rects = []
    for img in image_list:
        xref = img[0]
        # Βρες που εμφανίζεται η εικόνα στη σελίδα
        for item in page.get_image_rects(xref):
            rects.append((item.x0, item.y0, item.x1, item.y1))
    return rects


def _detect_columns(page: fitz.Page) -> Tuple[bool, Optional[int]]:
    """
    Renders τη σελίδα σε grayscale και ψάχνει για κατακόρυφη λευκή γραμμή
    που χωρίζει τη σελίδα σε δύο στήλες (συνήθης μορφή ΦΕΚ).
    """
    try:
        # Render σε μικρή ανάλυση για ταχύτητα
        mat = fitz.Matrix(0.5, 0.5)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)

        h, w = img.shape
        # Ψάχνουμε μόνο στο κεντρικό 40-60% του πλάτους
        x_start = int(w * COLUMN_SEARCH_BAND[0])
        x_end   = int(w * COLUMN_SEARCH_BAND[1])

        best_x    = None
        best_score = 0.0

        for x in range(x_start, x_end):
            col = img[:, x]
            white_ratio = np.sum(col > 240) / h
            if white_ratio > best_score:
                best_score = white_ratio
                best_x = x

        if best_score >= COLUMN_WHITE_RATIO:
            # Κλιμάκωση πίσω στις πραγματικές διαστάσεις (factor 0.5)
            return True, best_x * 2

        return False, None

    except Exception:
        return False, None
