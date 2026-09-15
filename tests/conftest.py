"""
tests/conftest.py — Shared fixtures for Receipt Intelligence tests.

All tests run WITHOUT a real Groq API key by mocking the GroqService.
"""

from __future__ import annotations

import pytest

from core.models import (
    LineItemValidation,
    Receipt,
    ReceiptItem,
    ReceiptValidation,
    ValidationStatus,
)


# ---------------------------------------------------------------------------
# Sample receipt data
# ---------------------------------------------------------------------------

SAMPLE_RECEIPT_DICT = {
    "merchant_name": "ABC Supermarket",
    "receipt_date": "2024-01-15",
    "currency": "$",
    "subtotal": 153.00,
    "tax": 12.24,
    "discount": None,
    "total": 165.24,
    "payment_method": "VISA",
    "items": [
        {
            "name": "Nike Running Shoes",
            "quantity": 1,
            "unit_price": 120.00,
            "total_price": 120.00,
            "is_uncertain": False,
            "extraction_confidence": 0.95,
            "original_text": "Nike Running Shoes    1    $120.00",
        },
        {
            "name": "Coca Cola 2L",
            "quantity": 2,
            "unit_price": 4.00,
            "total_price": 8.00,
            "is_uncertain": False,
            "extraction_confidence": 0.98,
            "original_text": "Coca Cola 2L    2    $4.00    $8.00",
        },
        {
            "name": "Apple iPhone Case",
            "quantity": 1,
            "unit_price": 25.00,
            "total_price": 25.00,
            "is_uncertain": False,
            "extraction_confidence": 0.92,
            "original_text": "Apple iPhone Case    1    $25.00",
        },
    ],
}


@pytest.fixture
def sample_items() -> list[ReceiptItem]:
    return [
        ReceiptItem(
            name="Nike Running Shoes",
            quantity=1,
            unit_price=120.00,
            total_price=120.00,
            extraction_confidence=0.95,
        ),
        ReceiptItem(
            name="Coca Cola 2L",
            quantity=2,
            unit_price=4.00,
            total_price=8.00,
            extraction_confidence=0.98,
        ),
        ReceiptItem(
            name="Apple iPhone Case",
            quantity=1,
            unit_price=25.00,
            total_price=25.00,
            extraction_confidence=0.92,
        ),
    ]


@pytest.fixture
def sample_receipt(sample_items) -> Receipt:
    return Receipt(
        merchant_name="ABC Supermarket",
        receipt_date="2024-01-15",
        currency="$",
        subtotal=153.00,
        tax=12.24,
        total=165.24,
        items=sample_items,
        ocr_text="ABC Supermarket\nNike Running Shoes 1 $120\nCoca Cola 2L 2 $8\nApple iPhone Case 1 $25\nSubtotal $153\nTax $12.24\nTotal $165.24",
        ocr_quality_score=0.90,
    )
