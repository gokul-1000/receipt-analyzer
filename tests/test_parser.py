"""
tests/test_parser.py — Tests for JSON parsing, normalisation, and confidence scoring.
"""

from __future__ import annotations

import pytest

from utils.parsing import (
    _extract_json_block,
    _strip_markdown_fences,
    parse_llm_json,
)
from utils.normalization import (
    normalize_currency,
    normalize_date,
    normalize_price,
    normalize_quantity,
)
from utils.confidence import estimate_ocr_quality, compute_structural_score
from core.models import ReceiptItem


# ---------------------------------------------------------------------------
# JSON parsing tests
# ---------------------------------------------------------------------------


class TestParseLLMJson:
    def test_clean_json_object(self):
        raw = '{"items": [], "total": 100}'
        result = parse_llm_json(raw)
        assert result == {"items": [], "total": 100}

    def test_json_in_markdown_fence(self):
        raw = '```json\n{"items": []}\n```'
        result = parse_llm_json(raw)
        assert result == {"items": []}

    def test_json_array(self):
        raw = '[{"category": "Food & Beverages", "confidence": 0.9}]'
        result = parse_llm_json(raw)
        assert isinstance(result, list)
        assert result[0]["category"] == "Food & Beverages"

    def test_json_with_trailing_comma(self):
        """Trailing commas are a common LLM output mistake."""
        raw = '{"items": [], "total": 100,}'
        result = parse_llm_json(raw)
        assert result is not None

    def test_json_with_preamble(self):
        """LLM sometimes adds text before the JSON."""
        raw = 'Here is the extracted data:\n\n{"items": [], "total": 50}'
        result = parse_llm_json(raw)
        assert result is not None and result.get("total") == 50

    def test_empty_string_returns_none(self):
        assert parse_llm_json("") is None
        assert parse_llm_json("   ") is None

    def test_invalid_json_returns_none(self):
        assert parse_llm_json("not json at all") is None

    def test_malformed_json_returns_none(self):
        assert parse_llm_json("{broken: json}") is None


# ---------------------------------------------------------------------------
# Normalisation tests
# ---------------------------------------------------------------------------


class TestNormalizePrice:
    def test_plain_number(self):
        assert normalize_price("100") == 100.0

    def test_dollar_string(self):
        assert normalize_price("$1,299.00") == 1299.0

    def test_rupee_string(self):
        assert normalize_price("₹1,299") == 1299.0

    def test_european_format(self):
        """European format: 1.299,00 = 1299.00"""
        assert normalize_price("1.299,00") == 1299.0

    def test_float_passthrough(self):
        assert normalize_price(49.99) == 49.99

    def test_integer_passthrough(self):
        assert normalize_price(100) == 100.0

    def test_none_returns_none(self):
        assert normalize_price(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_price("") is None

    def test_negative_returns_none(self):
        assert normalize_price(-5.0) is None

    def test_k_suffix(self):
        assert normalize_price("1.5k") == 1500.0


class TestNormalizeQuantity:
    def test_integer(self):
        assert normalize_quantity("3") == 3.0

    def test_float(self):
        assert normalize_quantity("1.5") == 1.5

    def test_zero_returns_none(self):
        assert normalize_quantity(0) is None

    def test_none_returns_none(self):
        assert normalize_quantity(None) is None


class TestNormalizeDate:
    def test_iso_format_unchanged(self):
        assert normalize_date("2024-01-15") == "2024-01-15"

    def test_slash_format(self):
        result = normalize_date("15/01/2024")
        assert result == "2024-01-15"

    def test_none_returns_none(self):
        assert normalize_date(None) is None

    def test_empty_returns_none(self):
        assert normalize_date("") is None

    def test_unknown_format_returned_as_is(self):
        result = normalize_date("Jan 2024")
        assert result == "Jan 2024"  # Returned unchanged, not None


class TestNormalizeCurrency:
    def test_iso_code_unchanged(self):
        assert normalize_currency("USD") == "USD"

    def test_symbol_accepted(self):
        result = normalize_currency("$")
        assert result == "$"

    def test_none_returns_none(self):
        assert normalize_currency(None) is None

    def test_empty_returns_none(self):
        assert normalize_currency("") is None

    def test_lowercase_iso(self):
        result = normalize_currency("usd")
        assert result == "USD"


# ---------------------------------------------------------------------------
# Confidence scoring tests
# ---------------------------------------------------------------------------


class TestOCRQuality:
    def test_empty_text_returns_zero(self):
        assert estimate_ocr_quality("") == 0.0
        assert estimate_ocr_quality("   ") == 0.0

    def test_good_text_returns_high_score(self):
        text = "ABC Supermarket\nNike Shoes 1 $120\nCoca Cola $8\nTotal $128"
        score = estimate_ocr_quality(text)
        assert score > 0.5

    def test_garbage_text_returns_low_score(self):
        text = "@@##!!** ??? ~~~"
        score = estimate_ocr_quality(text)
        assert score < 0.5


class TestStructuralScore:
    def test_complete_item_high_score(self):
        item = ReceiptItem(
            name="Nike Shoes",
            quantity=1,
            unit_price=120.0,
            total_price=120.0,
            category="Clothing & Footwear",
            category_confidence=0.95,
        )
        score = compute_structural_score(item)
        assert score >= 0.8

    def test_incomplete_item_low_score(self):
        item = ReceiptItem(
            name="Unknown",
            quantity=None,
            unit_price=None,
            total_price=None,
            category="Other / Uncategorized",
        )
        score = compute_structural_score(item)
        assert score < 0.5
