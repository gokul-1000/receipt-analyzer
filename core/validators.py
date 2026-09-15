"""
core/validators.py — Deterministic validation of extracted receipt data.

This module performs arithmetic and structural checks *without* calling
the LLM.  Results are attached to the Receipt model before it is returned
to the UI.
"""

from __future__ import annotations

import logging
from typing import Optional

from core.models import (
    LineItemValidation,
    Receipt,
    ReceiptItem,
    ReceiptValidation,
    ValidationStatus,
)
from config import (
    LINE_ITEM_TOLERANCE,
    SUBTOTAL_TOLERANCE,
    TOTAL_TOLERANCE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Line-item validation
# ---------------------------------------------------------------------------


def validate_line_item(item: ReceiptItem) -> LineItemValidation:
    """
    Check whether qty × unit_price ≈ total_price.

    Returns a LineItemValidation describing the result.
    Missing values are not flagged as errors – only clear mismatches are.
    """
    qty = item.quantity
    unit = item.unit_price
    total = item.total_price

    # If we have all three values, check arithmetic
    if qty is not None and unit is not None and total is not None:
        expected = round(qty * unit, 4)
        discrepancy = abs(expected - total)
        tolerance = max(LINE_ITEM_TOLERANCE * total, 0.01)  # at least 1 cent

        if discrepancy <= tolerance:
            return LineItemValidation(
                is_valid=True,
                expected_total=expected,
                actual_total=total,
                discrepancy=discrepancy,
                note="Line total matches qty × unit price.",
            )
        else:
            return LineItemValidation(
                is_valid=False,
                expected_total=expected,
                actual_total=total,
                discrepancy=discrepancy,
                note=(
                    f"Possible extraction error: {qty} × {unit} = {expected:.2f} "
                    f"but receipt shows {total:.2f} (diff: {discrepancy:.2f})"
                ),
            )

    # If only total_price is available → line is incomplete but not invalid
    if total is not None:
        return LineItemValidation(
            is_valid=True,
            actual_total=total,
            note="Qty or unit price missing; cannot verify arithmetic.",
        )

    return LineItemValidation(
        is_valid=True,
        note="No price information available for this line.",
    )


# ---------------------------------------------------------------------------
# Receipt-level validation
# ---------------------------------------------------------------------------


def validate_receipt(receipt: Receipt) -> ReceiptValidation:
    """
    Validate the entire receipt:

    1. Validate every line item.
    2. Check that Σ(line totals) ≈ subtotal (if available).
    3. Check that subtotal + tax - discount ≈ total (if available).

    Returns a ReceiptValidation attached to the receipt.
    """
    notes: list[str] = []
    line_items_ok = True
    subtotal_matches: Optional[bool] = None
    total_consistent: Optional[bool] = None
    tax_detected = receipt.tax is not None and receipt.tax > 0

    # --- 1. Validate each line item ---
    for item in receipt.items:
        lv = validate_line_item(item)
        item.line_validation = lv
        if not lv.is_valid:
            line_items_ok = False
            notes.append(f"⚠ Line mismatch — {item.name}: {lv.note}")

    # --- 2. Compare Σ(line totals) vs stated subtotal ---
    computed_sub = receipt.computed_total
    stated_sub = receipt.subtotal or receipt.total

    if computed_sub is not None and stated_sub is not None:
        tolerance = max(SUBTOTAL_TOLERANCE * stated_sub, 0.10)
        diff = abs(computed_sub - stated_sub)
        subtotal_matches = diff <= tolerance
        if subtotal_matches:
            notes.append(f"✓ Item totals sum ({computed_sub:.2f}) matches receipt subtotal ({stated_sub:.2f}).")
        else:
            notes.append(
                f"⚠ Item totals ({computed_sub:.2f}) differ from receipt subtotal ({stated_sub:.2f}) "
                f"by {diff:.2f}."
            )
    elif computed_sub is not None:
        notes.append(f"ℹ Computed item total: {computed_sub:.2f} (no stated subtotal to compare).")

    # --- 3. Check subtotal + tax - discount ≈ total ---
    if (
        receipt.subtotal is not None
        and receipt.total is not None
    ):
        reconstructed = receipt.subtotal
        if receipt.tax:
            reconstructed += receipt.tax
        if receipt.discount:
            reconstructed -= receipt.discount
        reconstructed = round(reconstructed, 4)
        total_diff = abs(reconstructed - receipt.total)
        tolerance = max(TOTAL_TOLERANCE * receipt.total, 0.10)
        total_consistent = total_diff <= tolerance
        if total_consistent:
            notes.append(
                f"✓ subtotal + tax − discount ({reconstructed:.2f}) matches total ({receipt.total:.2f})."
            )
        else:
            notes.append(
                f"⚠ Reconstructed total ({reconstructed:.2f}) differs from stated total "
                f"({receipt.total:.2f}) by {total_diff:.2f}."
            )

    # --- Determine overall validation status ---
    status = _determine_status(
        line_items_ok=line_items_ok,
        subtotal_matches=subtotal_matches,
        total_consistent=total_consistent,
        has_items=bool(receipt.items),
    )

    return ReceiptValidation(
        status=status,
        line_items_ok=line_items_ok,
        subtotal_matches=subtotal_matches if subtotal_matches is not None else False,
        tax_detected=tax_detected,
        total_consistent=total_consistent if total_consistent is not None else False,
        computed_subtotal=computed_sub,
        receipt_subtotal=receipt.subtotal,
        receipt_total=receipt.total,
        discrepancy=(
            abs((computed_sub or 0) - (stated_sub or 0))
            if computed_sub is not None and stated_sub is not None
            else None
        ),
        notes=notes,
    )


def _determine_status(
    line_items_ok: bool,
    subtotal_matches: Optional[bool],
    total_consistent: Optional[bool],
    has_items: bool,
) -> ValidationStatus:
    """Map individual check results to a top-level ValidationStatus."""

    if not has_items:
        return ValidationStatus.WARNING

    checks_done = [
        subtotal_matches,
        total_consistent,
    ]
    # Filter out None (checks that were impossible due to missing data)
    definitive = [c for c in checks_done if c is not None]

    all_pass = line_items_ok and all(definitive)
    any_fail = (not line_items_ok) or any(c is False for c in definitive)

    if all_pass and definitive:
        return ValidationStatus.VALIDATED
    if not definitive and line_items_ok:
        return ValidationStatus.PARTIALLY_VALIDATED
    if any_fail:
        if not line_items_ok and any(c is False for c in definitive):
            return ValidationStatus.FAILED_VALIDATION
        return ValidationStatus.WARNING
    return ValidationStatus.PARTIALLY_VALIDATED
