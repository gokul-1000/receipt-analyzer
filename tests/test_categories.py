"""
tests/test_categories.py — Tests for category validation and sanitisation.
"""

from __future__ import annotations

import pytest

from config import CATEGORY_LIST
from core.categories import (
    FALLBACK_CATEGORY,
    get_all_categories,
    get_category_description,
    is_valid_category,
    sanitize_category,
)


class TestCategoryValidation:
    def test_all_canonical_categories_valid(self):
        """Every category in CATEGORY_LIST should pass is_valid_category()."""
        for cat in CATEGORY_LIST:
            assert is_valid_category(cat), f"Category '{cat}' not recognised"

    def test_invalid_category_rejected(self):
        """Unknown categories should return False."""
        assert is_valid_category("Made Up Category") is False
        assert is_valid_category("") is False
        assert is_valid_category("food") is False

    def test_sanitize_returns_fallback_for_unknown(self):
        """Hallucinated categories should be replaced with fallback."""
        result = sanitize_category("Snacks & Treats")
        assert result == FALLBACK_CATEGORY

    def test_sanitize_preserves_valid_category(self):
        """Valid categories should pass through unchanged."""
        for cat in CATEGORY_LIST:
            assert sanitize_category(cat) == cat

    def test_sanitize_case_insensitive(self):
        """Case-insensitive match should succeed."""
        result = sanitize_category("food & beverages")
        assert result == "Food & Beverages"

    def test_fallback_category_in_list(self):
        """The fallback category must itself be a valid category."""
        assert is_valid_category(FALLBACK_CATEGORY)

    def test_get_all_categories_matches_config(self):
        """get_all_categories() must return exactly what's in config."""
        assert get_all_categories() == CATEGORY_LIST

    def test_descriptions_available_for_all(self):
        """Every category should have a non-empty description."""
        for cat in CATEGORY_LIST:
            desc = get_category_description(cat)
            assert desc, f"No description for category '{cat}'"


class TestCategoryEdgeCases:
    @pytest.mark.parametrize(
        "item_name, expected_category",
        [
            # These are just hints to verify taxonomy makes sense;
            # actual assignment is done by LLM — we test sanitize_category
            ("Other / Uncategorized", "Other / Uncategorized"),
            ("Food & Beverages", "Food & Beverages"),
            ("Electronics & Accessories", "Electronics & Accessories"),
        ],
    )
    def test_canonical_names_round_trip(self, item_name, expected_category):
        assert sanitize_category(item_name) == expected_category
