"""OCR helpers for image files and scanned PDF pages."""

import io
import os
import shutil
from pathlib import Path

from PIL import Image, ImageOps


def _configure_tesseract(pytesseract):
    """Configure Tesseract automatically on Windows/Linux when possible."""
    configured = os.environ.get("TESSERACT_CMD", "").strip()
    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
        return

    if shutil.which("tesseract"):
        return

    # Common Windows installation locations.
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Tesseract-OCR" / "tesseract.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Tesseract-OCR" / "tesseract.exe",
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return


def _load_pytesseract():
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is not installed. Run: pip install pytesseract"
        ) from exc

    _configure_tesseract(pytesseract)
    return pytesseract


def _prepare_image(image: Image.Image) -> Image.Image:
    """Light preprocessing that improves scanned-paper OCR."""
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    return image


def extract_text_from_image(data: bytes, *, psm: int = 6) -> str:
    """Extract text from PNG/JPG image bytes using Tesseract OCR."""
    pytesseract = _load_pytesseract()
    image = _prepare_image(Image.open(io.BytesIO(data)))
    try:
        return pytesseract.image_to_string(image, config=f"--psm {psm}")
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR engine was not found. Install Tesseract OCR and "
            "make sure it is in PATH, or set the TESSERACT_CMD environment variable."
        ) from exc


def extract_text_from_pdf_pages(
    document,
    *,
    dpi: int = 120,
    psm: int = 6,
) -> str:
    """OCR every page of a scanned PDF document and return combined text."""
    pytesseract = _load_pytesseract()
    page_text = []

    try:
        for page in document:
            pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            image = _prepare_image(image)
            text = pytesseract.image_to_string(image, config=f"--psm {psm}")
            if text.strip():
                page_text.append(text.strip())
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "This PDF appears to be scanned, so OCR is required, but Tesseract "
            "OCR was not found. Install Tesseract OCR and make sure it is in PATH, "
            "or set the TESSERACT_CMD environment variable."
        ) from exc

    return "\n\n".join(page_text)
