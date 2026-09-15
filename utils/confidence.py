"""
utils/confidence.py — Composite confidence scoring.

Confidence is NOT a single LLM-provided number.  It is computed from
multiple independent signals:

    1. OCR quality  (how readable was the source document?)
    2. Structural completeness  (how many key fields were extracted?)
    3. Arithmetic validation  (do the numbers add up?)
    4. Category confidence  (how certain is the categorisation?)

Each signal is normalised to [0, 1] and combined with configurable weights
defined in config.py.
"""

from __future__ import annotations

import logging
from typing import Optional

from config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_WEIGHT_ARITHMETIC,
    CONFIDENCE_WEIGHT_CATEGORY,
    CONFIDENCE_WEIGHT_OCR_QUALITY,
    CONFIDENCE_WEIGHT_STRUCTURAL,
)
from core.models import ConfidenceTier, Receipt, ReceiptItem, ValidationStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCR quality signal
# ---------------------------------------------------------------------------


def estimate_ocr_quality(text: str) -> float:
    """
    Heuristic estimate of OCR quality from the extracted text.

    Signals of high quality:
    - High ratio of printable alphanumeric characters
    - Presence of digits (prices, quantities)
    - Reasonable line length distribution

    Returns a score in [0, 1].
    """
    if not text or not text.strip():
        return 0.0

    chars = list(text)
    total = len(chars)
    if total == 0:
        return 0.0

    # Alphanumeric ratio
    alnum = sum(1 for c in chars if c.isalnum())
    alnum_ratio = alnum / total

    # Digit presence (receipts should have prices / dates)
    digit_count = sum(1 for c in chars if c.isdigit())
    has_digits = min(1.0, digit_count / max(1, total * 0.02))  # expect at least 2% digits

    # Line count sanity (extremely short → bad scan)
    lines = [l for l in text.splitlines() if l.strip()]
    line_score = min(1.0, len(lines) / 5)  # at least 5 meaningful lines → full score

    # Combine
    score = (
        0.50 * alnum_ratio
        + 0.30 * has_digits
        + 0.20 * line_score
    )
    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Structural completeness signal
# ---------------------------------------------------------------------------


def compute_structural_score(item: ReceiptItem) -> float:
    """
    Score based on how many key fields are populated for a single item.
    """
    fields = [
        item.name and item.name.strip(),
        item.total_price is not None,
        item.quantity is not None,
        item.unit_price is not None,
        item.category != "Other / Uncategorized",
    ]
    return round(sum(bool(f) for f in fields) / len(fields), 4)


def compute_receipt_structural_score(receipt: Receipt) -> float:
    """Overall structural completeness for a receipt."""
    fields = [
        receipt.merchant_name,
        receipt.receipt_date,
        receipt.total is not None,
        receipt.currency,
        bool(receipt.items),
    ]
    return round(sum(bool(f) for f in fields) / len(fields), 4)


# ---------------------------------------------------------------------------
# Arithmetic validation signal
# ---------------------------------------------------------------------------


def compute_arithmetic_score(item: ReceiptItem) -> float:
    """1.0 if line validated, 0.5 if unchecked, 0.0 if failed."""
    if item.line_validation.is_valid:
        if item.line_validation.expected_total is not None:
            return 1.0   # Full check passed
        return 0.7       # Unchecked (missing data) — neutral
    return 0.0


def compute_receipt_arithmetic_score(receipt: Receipt) -> float:
    """0-1 score from the receipt validation status."""
    status = receipt.validation.status
    mapping = {
        ValidationStatus.VALIDATED: 1.0,
        ValidationStatus.PARTIALLY_VALIDATED: 0.7,
        ValidationStatus.WARNING: 0.4,
        ValidationStatus.FAILED_VALIDATION: 0.0,
        ValidationStatus.NOT_CHECKED: 0.5,
    }
    return mapping.get(status, 0.5)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def compute_item_confidence(
    item: ReceiptItem,
    ocr_quality: float,
) -> float:
    """
    Compute and store the overall extraction confidence for a single item.

    The score is a weighted combination of:
    - OCR quality (shared across all items on the receipt)
    - Structural completeness of this item
    - Arithmetic consistency of this item
    - Category confidence provided by the LLM
    """
    structural = compute_structural_score(item)
    arithmetic = compute_arithmetic_score(item)
    category = item.category_confidence

    score = (
        CONFIDENCE_WEIGHT_OCR_QUALITY * ocr_quality
        + CONFIDENCE_WEIGHT_STRUCTURAL * structural
        + CONFIDENCE_WEIGHT_ARITHMETIC * arithmetic
        + CONFIDENCE_WEIGHT_CATEGORY * category
    )

    # Penalise uncertain items
    if item.is_uncertain:
        score *= 0.7

    return round(min(1.0, max(0.0, score)), 4)


def compute_receipt_confidence(receipt: Receipt) -> float:
    """
    Compute overall receipt-level confidence.
    """
    ocr_q = receipt.ocr_quality_score
    structural = compute_receipt_structural_score(receipt)
    arithmetic = compute_receipt_arithmetic_score(receipt)

    # Average category confidence across items
    if receipt.items:
        cat_conf = sum(i.category_confidence for i in receipt.items) / len(receipt.items)
    else:
        cat_conf = 0.3

    score = (
        CONFIDENCE_WEIGHT_OCR_QUALITY * ocr_q
        + CONFIDENCE_WEIGHT_STRUCTURAL * structural
        + CONFIDENCE_WEIGHT_ARITHMETIC * arithmetic
        + CONFIDENCE_WEIGHT_CATEGORY * cat_conf
    )
    return round(min(1.0, max(0.0, score)), 4)


def tier_label(score: float) -> str:
    """Return human-readable confidence tier."""
    if score >= CONFIDENCE_HIGH:
        return "🟢 High"
    if score >= CONFIDENCE_MEDIUM:
        return "🟡 Medium"
    return "🔴 Low"


def confidence_tier(score: float) -> ConfidenceTier:
    if score >= CONFIDENCE_HIGH:
        return ConfidenceTier.HIGH
    if score >= CONFIDENCE_MEDIUM:
        return ConfidenceTier.MEDIUM
    return ConfidenceTier.LOW
