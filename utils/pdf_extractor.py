"""PDF text extraction with automatic OCR fallback for scanned PDFs."""

import io

from utils.ocr import extract_text_from_pdf_pages
from utils.question_extractor import MAIN_Q


def _page_has_large_image(page):
    """Return True when a page appears to be a full-page scan image."""
    page_area = page.rect.get_area()
    if not page_area:
        return False

    image_area = 0.0
    for image in page.get_images(full=True):
        for rect in page.get_image_rects(image[0]):
            image_area += rect.get_area()
    return image_area / page_area >= 0.55


def extract_pdf_text(file_obj, *, ocr_fallback: bool = True, ocr_dpi: int = 200) -> str:
    """Extract selectable PDF text, falling back to OCR for scanned PDFs.

    The file object is read once. Pages with little selectable text are rendered
    and passed through Tesseract OCR, including pages in mixed text/scan PDFs.
    """
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is not installed. Run: pip install PyMuPDF") from exc

    data = file_obj.read()
    document = pymupdf.open(stream=data, filetype="pdf")

    # OCR pages with little text, no recognized question numbering, or a
    # full-page scan image. This also handles mixed text/scan PDFs.
    extracted_pages = []
    for page in document:
        page_text = page.get_text("text").strip()
        native_question_count = len(list(MAIN_Q.finditer(page_text)))
        needs_ocr = (
            len(page_text) < 60
            or native_question_count == 0
            or _page_has_large_image(page)
        )
        if ocr_fallback and needs_ocr:
            ocr_text = extract_text_from_pdf_pages([page], dpi=ocr_dpi, psm=6)
            ocr_question_count = len(list(MAIN_Q.finditer(ocr_text)))
            if ocr_question_count > native_question_count:
                page_text = ocr_text
            elif not page_text or len(page_text) < 60:
                page_text = ocr_text or page_text
            elif native_question_count == 0 and len(ocr_text) > len(page_text):
                page_text = ocr_text
        extracted_pages.append(page_text)

    return "\n\n".join(text for text in extracted_pages if text).strip()
