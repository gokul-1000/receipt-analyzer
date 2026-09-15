"""
services/pdf_service.py — PDF text extraction and page rendering.

Strategy:
1. Open the PDF with PyMuPDF (fitz).
2. Attempt to extract embedded/selectable text from each page.
3. If a page has very little selectable text (likely a scanned image),
   render that page to a high-DPI image and run OCR on it.
4. Combine text from all pages.

Technology choice: PyMuPDF (fitz)
Rationale:
- Zero external dependencies (pure Python wheel)
- Fast, accurate text extraction from native PDFs
- High-quality page rendering for scanned PDFs
- Better than pdf2image (which requires poppler binary)
- Works on Streamlit Cloud without additional system packages
"""

from __future__ import annotations

import io
import logging
from typing import Optional

from PIL import Image

from config import PDF_DPI, PDF_MAX_PAGES, MAX_FILE_SIZE_MB
from core.models import OCRResult
from utils.normalization import clean_ocr_text

logger = logging.getLogger(__name__)

try:
    import pymupdf as fitz  # PyMuPDF ≥ 1.24 prefers this import path
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz  # Older PyMuPDF versions
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False
        logger.warning("PyMuPDF not installed — PDF processing will not work.")


class PDFProcessingError(Exception):
    """Raised when PDF processing fails."""
    pass


# Minimum character count per page to consider it "selectable text"
_MIN_SELECTABLE_TEXT_CHARS = 50


def process_pdf(pdf_bytes: bytes) -> OCRResult:
    """
    Extract text from a PDF, using OCR on pages with no selectable text.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file content.

    Returns
    -------
    OCRResult
        Combined text from all pages.

    Raises
    ------
    PDFProcessingError
        If the PDF cannot be opened or processed.
    """
    if not PYMUPDF_AVAILABLE:
        raise PDFProcessingError(
            "PyMuPDF is not installed. Add 'pymupdf' to requirements.txt."
        )

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise PDFProcessingError(
            f"PDF exceeds maximum size of {MAX_FILE_SIZE_MB} MB."
        )

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFProcessingError(f"Failed to open PDF: {exc}") from exc

    page_count = min(len(doc), PDF_MAX_PAGES)
    if page_count == 0:
        raise PDFProcessingError("PDF contains no pages.")

    if page_count < len(doc):
        logger.warning(
            "PDF has %d pages; processing only the first %d.",
            len(doc), page_count,
        )

    all_text_parts: list[str] = []
    quality_scores: list[float] = []
    warnings: list[str] = []
    is_selectable = True  # Will be set False if any page needs OCR

    for page_idx in range(page_count):
        page = doc[page_idx]
        page_text = page.get_text("text").strip()

        if len(page_text) >= _MIN_SELECTABLE_TEXT_CHARS:
            # Good: native text available
            all_text_parts.append(page_text)
            quality_scores.append(0.90)  # Native text is high quality
            logger.debug("Page %d: used selectable text (%d chars)", page_idx + 1, len(page_text))
        else:
            # Scanned page → render to image and OCR
            is_selectable = False
            logger.debug("Page %d: no selectable text — rasterising for OCR", page_idx + 1)
            try:
                ocr_result = _ocr_page(page)
                all_text_parts.append(ocr_result.text)
                quality_scores.append(ocr_result.quality_score)
                warnings.extend(ocr_result.warnings)
            except Exception as exc:
                logger.warning("Page %d OCR failed: %s", page_idx + 1, exc)
                warnings.append(f"Page {page_idx + 1}: OCR failed — {exc}")

    doc.close()

    combined_text = "\n\n--- PAGE BREAK ---\n\n".join(all_text_parts)
    combined_text = clean_ocr_text(combined_text)

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.3

    return OCRResult(
        text=combined_text,
        quality_score=round(avg_quality, 4),
        page_count=page_count,
        is_selectable_text=is_selectable,
        method_used="pymupdf" if is_selectable else "pymupdf+tesseract",
        warnings=warnings,
    )


def _ocr_page(page: "fitz.Page") -> OCRResult:
    """
    Render a single PDF page to an image and run Tesseract OCR on it.
    """
    from services.ocr_service import run_ocr  # Import here to avoid circular import

    # Render at PDF_DPI
    mat = fitz.Matrix(PDF_DPI / 72, PDF_DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    pil_image = Image.open(io.BytesIO(img_bytes))
    return run_ocr(pil_image)


def render_pdf_page_preview(pdf_bytes: bytes, page_idx: int = 0) -> Optional[Image.Image]:
    """
    Render a single PDF page to a PIL Image for display in the UI.

    Returns None on failure (non-fatal).
    """
    if not PYMUPDF_AVAILABLE:
        return None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if page_idx >= len(doc):
            return None
        page = doc[page_idx]
        mat = fitz.Matrix(1.5, 1.5)  # 108 DPI — good for preview
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        doc.close()
        return Image.open(io.BytesIO(img_bytes))
    except Exception as exc:
        logger.warning("PDF preview render failed: %s", exc)
        return None


def validate_pdf_file(data: bytes) -> tuple[bool, str]:
    """
    Validate that the uploaded bytes represent a valid PDF.

    Returns (is_valid, error_message).
    """
    from config import MAX_FILE_SIZE_MB
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(data) > max_bytes:
        return False, f"File exceeds maximum size of {MAX_FILE_SIZE_MB} MB."

    # Check PDF magic bytes
    if not data[:4] == b"%PDF":
        return False, "File does not appear to be a valid PDF."

    if not PYMUPDF_AVAILABLE:
        return True, ""  # Can't verify further without PyMuPDF

    try:
        doc = fitz.open(stream=data, filetype="pdf")
        page_count = len(doc)
        doc.close()
        if page_count == 0:
            return False, "PDF contains no pages."
        return True, ""
    except Exception as exc:
        return False, f"PDF is corrupted or unreadable: {exc}"
