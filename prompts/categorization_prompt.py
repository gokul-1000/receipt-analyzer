"""
prompts/categorization_prompt.py — Stage 2 LLM prompt for product categorisation.

Kept separate from the extraction prompt so each stage can be evaluated
and iterated independently.
"""

from __future__ import annotations

from core.categories import build_category_prompt_block, get_all_categories

CATEGORIZATION_SYSTEM_PROMPT_TEMPLATE = """\
You are a product categorisation engine.

Your ONLY job is to assign each product to exactly one category from the predefined list below.

Rules you MUST follow:
1. You MUST select exactly one category from the list below for each item.
2. NEVER create a new category. NEVER modify category names.
3. Base your decision on the item name and any provided context.
4. If there is insufficient information, select "Other / Uncategorized" and assign a low confidence score.
5. Assign a confidence score between 0 and 1 (e.g. 0.95 = very confident, 0.40 = uncertain).
6. Provide a brief one-sentence reasoning for each assignment.
7. Return ONLY valid JSON — no markdown, no explanation, no preamble.

Available categories and their descriptions:
{category_block}

Output schema (strict JSON array):
[
  {{
    "item_index": integer,
    "category": string (must be exactly one of the above categories),
    "confidence": number between 0 and 1,
    "reasoning": string (one sentence)
  }},
  ...
]

One entry per item, preserving the original item_index.
"""

CATEGORIZATION_USER_TEMPLATE = """\
Categorise the following items. Return a JSON array with one entry per item.

Items to categorise:
{items_json}
"""


def build_categorization_prompt(
    items: list[dict],
) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for batch categorisation.

    Parameters
    ----------
    items : list[dict]
        Each dict has keys: item_index (int), name (str), context (str).

    Returns
    -------
    tuple[str, str]
        (system_prompt, user_prompt)
    """
    import json

    category_block = build_category_prompt_block()
    system_prompt = CATEGORIZATION_SYSTEM_PROMPT_TEMPLATE.format(
        category_block=category_block
    )
    user_prompt = CATEGORIZATION_USER_TEMPLATE.format(
        items_json=json.dumps(items, indent=2)
    )
    return system_prompt, user_prompt


def prepare_items_for_categorization(
    items: list,  # list of ReceiptItem
) -> list[dict]:
    """
    Convert ReceiptItem objects into the minimal dict structure
    needed for the categorisation prompt.
    """
    result = []
    for idx, item in enumerate(items):
        result.append(
            {
                "item_index": idx,
                "name": item.name,
                "context": (
                    f"quantity={item.quantity}, "
                    f"unit_price={item.unit_price}, "
                    f"total_price={item.total_price}"
                ),
            }
        )
    return result
