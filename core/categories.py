"""
core/categories.py — Category taxonomy helpers.

The canonical category list lives in config.py.  This module provides
helper functions used by the categorisation service and UI.
"""

from __future__ import annotations

from config import CATEGORY_LIST, CATEGORY_DESCRIPTIONS

# Keep a frozenset for O(1) membership checks
_VALID_CATEGORIES: frozenset[str] = frozenset(CATEGORY_LIST)

FALLBACK_CATEGORY: str = "Other / Uncategorized"


def is_valid_category(category: str) -> bool:
    """Return True if *category* is in the predefined taxonomy."""
    return category in _VALID_CATEGORIES


def sanitize_category(category: str) -> str:
    """
    Return *category* unchanged if valid; otherwise return the fallback.

    This is the primary guard against LLM hallucinating new categories.
    """
    if is_valid_category(category):
        return category
    # Try case-insensitive match as a courtesy
    lower = category.strip().lower()
    for valid in _VALID_CATEGORIES:
        if valid.lower() == lower:
            return valid
    return FALLBACK_CATEGORY


def get_category_description(category: str) -> str:
    """Return the human-readable description for a category."""
    return CATEGORY_DESCRIPTIONS.get(category, "")


def get_all_categories() -> list[str]:
    """Return the ordered category list (preserves taxonomy order)."""
    return list(CATEGORY_LIST)


def build_category_prompt_block() -> str:
    """
    Build a formatted string listing all categories with their descriptions
    for insertion into LLM prompts.
    """
    lines: list[str] = []
    for cat in CATEGORY_LIST:
        desc = CATEGORY_DESCRIPTIONS.get(cat, "")
        lines.append(f'- "{cat}": {desc}')
    return "\n".join(lines)
