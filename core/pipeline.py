"""
core/pipeline.py — Main receipt processing pipeline orchestrator.

This module wires all services together into a single process_receipt()
function.  Each stage is clearly separated and can be replaced/mocked
independently.

Pipeline stages:
    1. File validation
    2. OCR / text extraction
    3. Stage 1 LLM extraction (receipt metadata + items)
    4. Deterministic validation
    5. Stage 2 batch categorisation
    6. Confidence scoring
    7. Final assembly into Receipt model
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from PIL import Image

from config import (
    GROQ_MAX_TOKENS_CATEGORIZATION,
    GROQ_MAX_TOKENS_EXTRACTION,
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_PDF_TYPES,
)
from core.models import OCRResult, Receipt, ReceiptItem
from core.validators import validate_receipt
from prompts.extraction_prompt import (
    build_extraction_prompt,
    build_vision_extraction_prompt,
)
from services.categorization_service import categorize_receipt_items
from services.groq_service import GroqService, GroqServiceError
from services.image_service import (
    ImageProcessingError,
    load_image,
    prepare_image_for_vision,
    validate_image_file,
)
from services.ocr_service import OCRError, run_ocr
from services.pdf_service import (
    PDFProcessingError,
    process_pdf,
    render_pdf_page_preview,
    validate_pdf_file,
)
from utils.confidence import (
    compute_item_confidence,
    compute_receipt_confidence,
    estimate_ocr_quality,
)
from utils.normalization import (
    normalize_currency,
    normalize_date,
    normalize_price,
    normalize_quantity,
)
from utils.parsing import ensure_list, parse_llm_json, safe_get

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when the pipeline cannot produce a result."""
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_receipt(
    file_bytes: bytes,
    filename: str,
    groq: GroqService,
    extraction_mode: str = "vision",
) -> Receipt:
    """
    Process a single receipt file through the pipeline.

    Parameters
    ----------
    file_bytes : bytes
        Raw file content.
    filename : str
        Original filename (used to detect file type).
    groq : GroqService
        Initialised Groq service instance.
    extraction_mode : str
        "vision" for direct Vision AI (qwen/qwen3.6-27b),
        or "ocr_llm" for traditional Tesseract OCR + LLM.

    Returns
    -------
    Receipt
        Fully populated Receipt model.

    Raises
    ------
    PipelineError
        On unrecoverable errors (validation failures, etc.).
    """
    start_time = time.time()
    ext = _get_extension(filename)

    logger.info(
        "Processing receipt: %s (mode=%s, ext=%s, size=%d bytes)",
        filename, extraction_mode, ext, len(file_bytes),
    )

    # ------------------------------------------------------------------ #
    # Stage 1: File validation                                            #
    # ------------------------------------------------------------------ #
    _validate_file(file_bytes, filename, ext)

    receipt: Optional[Receipt] = None
    fallback_note: Optional[str] = None
    used_mode = extraction_mode

    # ------------------------------------------------------------------ #
    # Stage 2 & 3: Extraction (Vision AI or OCR + LLM)                    #
    # ------------------------------------------------------------------ #
    if extraction_mode == "vision":
        try:
            receipt = _vision_extract(file_bytes, filename, ext, groq)
        except Exception as exc:
            logger.warning(
                "Vision AI extraction failed: %s. Falling back to OCR + LLM.",
                exc,
            )
            used_mode = "ocr_llm"
            fallback_note = f"ℹ Vision AI extraction failed ({exc}); fell back to OCR + LLM."
            receipt = None

    if receipt is None:
        # Standard OCR + LLM pipeline (primary if mode="ocr_llm", or fallback)
        ocr_result = _extract_text(file_bytes, filename, ext)
        logger.info(
            "OCR complete: %d chars, quality=%.2f",
            len(ocr_result.text), ocr_result.quality_score,
        )

        if not ocr_result.text.strip():
            raise PipelineError(
                "No text could be extracted from this file. "
                "Try uploading a clearer image or a higher-quality scan."
            )

        receipt = _llm_extract(ocr_result, groq, filename)
        receipt.ocr_text = ocr_result.text
        receipt.ocr_quality_score = ocr_result.quality_score
        receipt.page_count = ocr_result.page_count
        receipt.filename = filename

    # ------------------------------------------------------------------ #
    # Stage 4: Deterministic validation                                   #
    # ------------------------------------------------------------------ #
    receipt.validation = validate_receipt(receipt)
    if fallback_note:
        receipt.validation.notes.insert(0, fallback_note)
    logger.info("Validation status: %s", receipt.validation.status)

    # ------------------------------------------------------------------ #
    # Stage 5: Categorisation                                             #
    # ------------------------------------------------------------------ #
    if receipt.items:
        receipt = categorize_receipt_items(
            receipt,
            groq,
            max_tokens=GROQ_MAX_TOKENS_CATEGORIZATION,
        )

    # ------------------------------------------------------------------ #
    # Stage 6: Confidence scoring                                         #
    # ------------------------------------------------------------------ #
    for item in receipt.items:
        item.extraction_confidence = compute_item_confidence(
            item, ocr_quality=receipt.ocr_quality_score
        )

    receipt.overall_confidence = compute_receipt_confidence(receipt)
    receipt.processing_time_seconds = round(time.time() - start_time, 2)

    logger.info(
        "Pipeline complete: %d items, confidence=%.2f, time=%.2fs (mode=%s)",
        receipt.item_count,
        receipt.overall_confidence,
        receipt.processing_time_seconds,
        used_mode,
    )
    return receipt


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def _validate_file(data: bytes, filename: str, ext: str) -> None:
    """Raise PipelineError if the file is invalid."""
    if ext in SUPPORTED_IMAGE_TYPES:
        ok, msg = validate_image_file(data, filename)
    elif ext in SUPPORTED_PDF_TYPES:
        ok, msg = validate_pdf_file(data)
    else:
        raise PipelineError(
            f"Unsupported file type: .{ext}. "
            f"Please upload one of: {', '.join(SUPPORTED_IMAGE_TYPES + SUPPORTED_PDF_TYPES)}."
        )
    if not ok:
        raise PipelineError(msg)


def _extract_text(data: bytes, filename: str, ext: str) -> OCRResult:
    """Run OCR or text extraction depending on file type."""
    try:
        if ext in SUPPORTED_PDF_TYPES:
            return process_pdf(data)
        else:
            image = load_image(data)
            return run_ocr(image)
    except (OCRError, PDFProcessingError, ImageProcessingError) as exc:
        raise PipelineError(str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during text extraction: %s", exc)
        raise PipelineError(
            "An unexpected error occurred during text extraction. "
            "Please try a different file."
        ) from exc


def _llm_extract(ocr_result: OCRResult, groq: GroqService, filename: str) -> Receipt:
    """
    Call the LLM to extract structured receipt data from OCR text.

    Parses the response into a Receipt model with basic normalisation.
    """
    system_prompt, user_prompt = build_extraction_prompt(ocr_result.text)

    try:
        raw_response = groq.extract_receipt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=GROQ_MAX_TOKENS_EXTRACTION,
        )
    except GroqServiceError as exc:
        raise PipelineError(
            f"AI extraction failed: {exc}. Please check your Groq API key and try again."
        ) from exc

    parsed = parse_llm_json(raw_response)
    if not parsed or not isinstance(parsed, dict):
        raise PipelineError(
            "The AI returned an unexpected response format. "
            "This may be a temporary issue — please try again."
        )

    return _build_receipt_from_dict(parsed, ocr_result)


def _vision_extract(
    file_bytes: bytes,
    filename: str,
    ext: str,
    groq: GroqService,
) -> Receipt:
    """
    Direct multimodal receipt extraction using Groq Vision AI.
    """
    if ext in SUPPORTED_PDF_TYPES:
        preview_img = render_pdf_page_preview(file_bytes, page_idx=0)
        if preview_img is None:
            raise PipelineError("Could not render PDF page for Vision AI.")
        base64_img, mime_type = prepare_image_for_vision(preview_img)
    else:
        base64_img, mime_type = prepare_image_for_vision(file_bytes)

    system_prompt, user_prompt = build_vision_extraction_prompt()

    try:
        raw_response = groq.extract_receipt_vision(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_base64=base64_img,
            mime_type=mime_type,
            vision_model=groq.vision_model,
        )
    except GroqServiceError as exc:
        raise PipelineError(f"Groq Vision API failed: {exc}") from exc

    parsed = parse_llm_json(raw_response)
    if not parsed or not isinstance(parsed, dict):
        raise PipelineError("Vision AI returned an unexpected response format.")

    receipt = _build_receipt_from_dict(parsed, None)
    receipt.filename = filename
    receipt.ocr_text = _format_vision_summary(receipt, groq.vision_model)
    receipt.ocr_quality_score = 0.95
    receipt.page_count = 1
    return receipt


def _format_vision_summary(receipt: Receipt, model_name: str) -> str:
    """Generate a clean readable text log of the vision extraction."""
    lines = [
        f"[Direct Vision AI Extraction — {model_name}]",
        f"Merchant: {receipt.merchant_name or 'N/A'}",
        f"Date: {receipt.receipt_date or 'N/A'}",
        f"Currency: {receipt.currency or 'N/A'}",
        f"Total: {receipt.total or 'N/A'}",
        "",
        "Detected Line Items:",
    ]
    for item in receipt.items:
        lines.append(
            f"  • {item.name} | qty: {item.quantity or 'N/A'} | "
            f"unit: {item.unit_price or 'N/A'} | total: {item.total_price or 'N/A'}"
        )
    return "\n".join(lines)


def _build_receipt_from_dict(data: dict, ocr_result: Optional[OCRResult] = None) -> Receipt:
    """Convert the raw LLM JSON dict into a validated Receipt model."""

    # --- Normalise scalar fields ---
    currency = normalize_currency(safe_get(data, "currency"))
    if not currency and ocr_result and ocr_result.text:
        # Try to detect from OCR text
        from utils.normalization import extract_currency_from_text
        currency = extract_currency_from_text(ocr_result.text)

    receipt = Receipt(
        merchant_name=safe_get(data, "merchant_name") or None,
        receipt_date=normalize_date(safe_get(data, "receipt_date")),
        currency=currency,
        subtotal=normalize_price(safe_get(data, "subtotal")),
        tax=normalize_price(safe_get(data, "tax")),
        discount=normalize_price(safe_get(data, "discount")),
        total=normalize_price(safe_get(data, "total")),
        payment_method=safe_get(data, "payment_method") or None,
    )

    # --- Parse items ---
    raw_items = ensure_list(safe_get(data, "items", default=[]))
    receipt.items = [
        _build_item(raw_item)
        for raw_item in raw_items
        if isinstance(raw_item, dict) and raw_item.get("name")
    ]

    return receipt


def _build_item(raw: dict) -> ReceiptItem:
    """Parse a single item dict from LLM output into a ReceiptItem."""
    name = str(raw.get("name", "Unknown Item")).strip()
    if not name:
        name = "Unknown Item"

    qty = normalize_quantity(raw.get("quantity"))
    unit = normalize_price(raw.get("unit_price"))
    total = normalize_price(raw.get("total_price"))

    # If unit_price missing but we have qty and total, derive unit_price
    # ONLY if explicitly computable (not hallucinated)
    if unit is None and qty is not None and qty > 0 and total is not None:
        derived_unit = round(total / qty, 4)
        unit = derived_unit

    raw_conf = raw.get("extraction_confidence", 0.7)
    try:
        ext_conf = max(0.0, min(1.0, float(raw_conf)))
    except (TypeError, ValueError):
        ext_conf = 0.7

    is_uncertain = bool(raw.get("is_uncertain", False))
    original_text = str(raw.get("original_text", ""))

    return ReceiptItem(
        name=name,
        quantity=qty,
        unit_price=unit,
        total_price=total,
        is_uncertain=is_uncertain,
        extraction_confidence=ext_conf,
        original_text=original_text,
        # category will be filled in Stage 5
        category="Other / Uncategorized",
        category_confidence=0.5,
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _get_extension(filename: str) -> str:
    """Return lowercase file extension without the leading dot."""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()
