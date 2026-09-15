"""
tests/test_validator.py — Tests for arithmetic validation logic.
"""

from __future__ import annotations

import pytest

from core.models import Receipt, ReceiptItem, ValidationStatus
from core.validators import validate_line_item, validate_receipt


class TestLineItemValidation:
    def test_valid_arithmetic(self):
        """qty × unit_price = total_price → valid."""
        item = ReceiptItem(name="Test Item", quantity=2, unit_price=5.0, total_price=10.0)
        result = validate_line_item(item)
        assert result.is_valid is True

    def test_invalid_arithmetic(self):
        """qty × unit_price ≠ total_price → invalid."""
        item = ReceiptItem(name="Test Item", quantity=2, unit_price=5.0, total_price=11.0)
        result = validate_line_item(item)
        assert result.is_valid is False
        assert result.discrepancy is not None and result.discrepancy > 0

    def test_rounding_tolerance(self):
        """Small rounding differences should be tolerated."""
        item = ReceiptItem(name="Test Item", quantity=3, unit_price=1.99, total_price=5.97)
        result = validate_line_item(item)
        assert result.is_valid is True

    def test_missing_quantity(self):
        """Missing qty should not be flagged as invalid."""
        item = ReceiptItem(name="Test Item", quantity=None, unit_price=5.0, total_price=5.0)
        result = validate_line_item(item)
        assert result.is_valid is True

    def test_missing_unit_price(self):
        """Missing unit_price should not be flagged as invalid."""
        item = ReceiptItem(name="Test Item", quantity=2, unit_price=None, total_price=10.0)
        result = validate_line_item(item)
        assert result.is_valid is True

    def test_all_missing(self):
        """All prices missing → not invalid, just unchecked."""
        item = ReceiptItem(name="Test Item", quantity=None, unit_price=None, total_price=None)
        result = validate_line_item(item)
        assert result.is_valid is True

    def test_zero_quantity(self):
        """Zero quantity with zero total → valid edge case."""
        item = ReceiptItem(name="Test Item", quantity=1, unit_price=0.0, total_price=0.0)
        result = validate_line_item(item)
        assert result.is_valid is True


class TestReceiptValidation:
    def test_fully_valid_receipt(self, sample_receipt):
        """All items valid and totals match → VALIDATED."""
        result = validate_receipt(sample_receipt)
        assert result.line_items_ok is True

    def test_subtotal_mismatch(self, sample_items):
        """Incorrect subtotal should trigger a warning."""
        receipt = Receipt(
            subtotal=999.00,  # Wrong
            total=999.00,
            items=sample_items,
        )
        result = validate_receipt(receipt)
        assert result.subtotal_matches is False

    def test_no_items(self):
        """Receipt with no items → WARNING."""
        receipt = Receipt(total=100.0, items=[])
        result = validate_receipt(receipt)
        assert result.status == ValidationStatus.WARNING

    def test_tax_detected(self, sample_receipt):
        """Tax field should be detected."""
        result = validate_receipt(sample_receipt)
        assert result.tax_detected is True

    def test_no_tax(self, sample_items):
        """No tax → tax_detected should be False."""
        receipt = Receipt(items=sample_items, total=153.0, tax=None)
        result = validate_receipt(receipt)
        assert result.tax_detected is False

    def test_validated_status(self, sample_receipt):
        """Correct receipt → VALIDATED or PARTIALLY_VALIDATED."""
        result = validate_receipt(sample_receipt)
        assert result.status in (
            ValidationStatus.VALIDATED,
            ValidationStatus.PARTIALLY_VALIDATED,
        )
