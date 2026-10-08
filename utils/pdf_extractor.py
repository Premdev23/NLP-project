"""PDF text extraction with automatic OCR fallback for scanned PDFs."""

import io

from utils.ocr import extract_text_from_pdf_pages


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

    # OCR only pages with very little extractable text. This also handles PDFs
    # that mix selectable text pages with scanned pages.
    extracted_pages = []
    for page in document:
        page_text = page.get_text("text").strip()
        if ocr_fallback and len(page_text) < 60:
            ocr_text = extract_text_from_pdf_pages([page], dpi=ocr_dpi, psm=6)
            page_text = ocr_text or page_text
        extracted_pages.append(page_text)

    return "\n\n".join(text for text in extracted_pages if text).strip()
