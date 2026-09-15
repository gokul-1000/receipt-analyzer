"""
services/categorization_service.py — Two-stage categorisation orchestrator.

Calls Groq to categorise all extracted items in a SINGLE batch request
(not one request per item), then applies deterministic post-processing
to ensure all categories are from the predefined taxonomy.
"""

from __future__ import annotations

import logging
from typing import Optional

from core.categories import sanitize_category
from core.models import Receipt, ReceiptItem
from prompts.categorization_prompt import (
    build_categorization_prompt,
    prepare_items_for_categorization,
)
from services.groq_service import GroqService, GroqServiceError
from utils.parsing import ensure_list, parse_llm_json, safe_get

logger = logging.getLogger(__name__)


class CategorizationError(Exception):
    """Raised when categorisation fails completely."""
    pass


def categorize_receipt_items(
    receipt: Receipt,
    groq: GroqService,
    *,
    max_tokens: int = 2048,
) -> Receipt:
    """
    Categorise all items in *receipt* using a single Groq batch call.

    Modifies items in-place and returns the receipt.

    Parameters
    ----------
    receipt : Receipt
        The receipt whose items need categorisation.
    groq : GroqService
        Initialised Groq service.
    max_tokens : int
        Max tokens for the categorisation response.

    Returns
    -------
    Receipt
        Same receipt with category fields populated on each item.
    """
    if not receipt.items:
        return receipt

    # Build prompt
    items_for_prompt = prepare_items_for_categorization(receipt.items)
    system_prompt, user_prompt = build_categorization_prompt(items_for_prompt)

    # Call Groq
    try:
        raw_response = groq.categorize_items(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
    except GroqServiceError as exc:
        logger.error("Categorisation Groq call failed: %s", exc)
        # Apply fallback categories to all items
        _apply_fallback_categories(receipt.items)
        return receipt

    # Parse response
    parsed = parse_llm_json(raw_response)
    if not isinstance(parsed, list):
        logger.warning(
            "Categorisation response is not a list: %s", type(parsed)
        )
        _apply_fallback_categories(receipt.items)
        return receipt

    # Apply results
    _apply_category_results(receipt.items, parsed)
    return receipt


def _apply_category_results(
    items: list[ReceiptItem],
    results: list[dict],
) -> None:
    """
    Apply categorisation results from the LLM to the item list.

    For each result:
    - Sanitise the category (guard against hallucinated categories)
    - Apply confidence and reasoning
    - Fall back gracefully for missing/invalid entries
    """
    # Index results by item_index for O(1) lookup
    result_map: dict[int, dict] = {}
    for result in ensure_list(results):
        if isinstance(result, dict):
            idx = result.get("item_index")
            if isinstance(idx, int):
                result_map[idx] = result

    for i, item in enumerate(items):
        result = result_map.get(i)
        if not result:
            logger.debug("No category result for item %d (%s)", i, item.name)
            _apply_fallback_to_item(item)
            continue

        raw_category = safe_get(result, "category", default="Other / Uncategorized")
        category = sanitize_category(str(raw_category))

        raw_confidence = result.get("confidence", 0.5)
        try:
            confidence = max(0.0, min(1.0, float(raw_confidence)))
        except (TypeError, ValueError):
            confidence = 0.5

        # If the category had to be corrected, reduce confidence
        if category != raw_category:
            logger.warning(
                "Item %d: invalid category '%s' → corrected to '%s'",
                i, raw_category, category,
            )
            confidence = min(confidence, 0.4)

        reasoning = str(result.get("reasoning", ""))

        item.category = category
        item.category_confidence = confidence
        item.category_reasoning = reasoning


def _apply_fallback_categories(items: list[ReceiptItem]) -> None:
    for item in items:
        _apply_fallback_to_item(item)


def _apply_fallback_to_item(item: ReceiptItem) -> None:
    item.category = "Other / Uncategorized"
    item.category_confidence = 0.0
    item.category_reasoning = "Categorisation failed — defaulting to Other / Uncategorized."
