"""
OCR Engine - Tesseract για scanned σελίδες.
Χειρίζεται επίσης mixed σελίδες (scanned εικόνες μέσα σε text PDF)
και δύο στήλες.
"""
import fitz
import pytesseract
from PIL import Image
import numpy as np
import io
from typing import Optional, List
from .detector import PageInfo
from config import TESSERACT_LANG


# Tesseract config: OSD + Greek, PSM 3 = fully automatic page segmentation
TESS_CONFIG = f"--oem 1 --psm 3 -l {TESSERACT_LANG}"
# Για δύο στήλες: PSM 1 = automatic with OSD
TESS_CONFIG_COLUMNS = f"--oem 1 --psm 1 -l {TESSERACT_LANG}"


def ocr_page(page: fitz.Page, info: PageInfo, dpi: int = 300) -> str:
    """
    Κάνει OCR μία scanned (ή mixed) σελίδα με Tesseract.
    Επιστρέφει το εξαγόμενο κείμενο.
    """
    if info.has_two_columns and info.column_x:
        return _ocr_two_columns(page, info.column_x, dpi)
    else:
        return _ocr_full_page(page, dpi)


def _render_page(page: fitz.Page, dpi: int) -> Image.Image:
    """
    Renders PDF page σε PIL Image για Tesseract.
    """
    zoom = dpi / 72  # 72 = default PDF dpi
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))


def _preprocess(image: Image.Image) -> Image.Image:
    """
    Βασική επεξεργασία εικόνας για καλύτερο OCR:
    - Binarization (threshold)
    - Αφαίρεση noise
    """
    try:
        import cv2
        img_np = np.array(image)
        # Adaptive threshold για ανομοιόμορφο φωτισμό (scans)
        binary = cv2.adaptiveThreshold(
            img_np, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10
        )
        return Image.fromarray(binary)
    except ImportError:
        # Χωρίς cv2: απλό L mode
        return image.convert("L")


def _ocr_full_page(page: fitz.Page, dpi: int) -> str:
    """
    OCR ολόκληρης σελίδας.
    """
    img = _render_page(page, dpi)
    img = _preprocess(img)
    text = pytesseract.image_to_string(img, config=TESS_CONFIG)
    return text.strip()


def _ocr_two_columns(page: fitz.Page, column_x: int, dpi: int) -> str:
    """
    Χωρίζει σε δύο στήλες, κάνει OCR χωριστά, ενώνει.
    """
    rect = page.rect
    zoom = dpi / 72

    # Αριστερή στήλη
    left_rect  = fitz.Rect(rect.x0, rect.y0, column_x, rect.y1)
    right_rect = fitz.Rect(column_x, rect.y0, rect.x1, rect.y1)

    def ocr_region(clip_rect):
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect, colorspace=fitz.csGRAY)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img = _preprocess(img)
        return pytesseract.image_to_string(img, config=TESS_CONFIG_COLUMNS).strip()

    left_text  = ocr_region(left_rect)
    right_text = ocr_region(right_rect)

    parts = []
    if left_text:
        parts.append(left_text)
    if right_text:
        parts.append(right_text)

    return "\n\n--- ΣΤΗΛΗ 2 ---\n\n".join(parts)


def ocr_document(pdf_path: str, page_infos: List[PageInfo],
                 progress_callback=None) -> List[dict]:
    """
    Κάνει OCR σε όλες τις scanned σελίδες ενός PDF.
    progress_callback(page_num, total) για live updates.
    """
    doc = fitz.open(pdf_path)
    results = []
    scanned_pages = [p for p in page_infos if p.is_scanned or p.is_mixed]
    total = len(scanned_pages)

    for i, info in enumerate(scanned_pages):
        if progress_callback:
            progress_callback(i + 1, total, info.page_num)

        page = doc[info.page_num - 1]
        text = ocr_page(page, info)

        results.append({
            "page_num": info.page_num,
            "text": text,
            "method": "tesseract"
        })

    doc.close()
    return results