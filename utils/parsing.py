"""
utils/parsing.py — Robust JSON parsing helpers for LLM output.

LLMs sometimes emit JSON wrapped in markdown fences, with trailing commas,
or with small structural errors.  This module attempts to recover valid JSON
from imperfect model output without silently swallowing errors.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_llm_json(raw: str) -> Optional[dict | list]:
    """
    Attempt to extract and parse JSON from raw LLM output.

    Strategy:
    1. Try to parse the full string directly.
    2. Extract content between the first { and matching } (or [ … ]).
    3. Strip markdown code fences and retry.
    4. Return None if all attempts fail (caller decides how to handle).
    """
    if not raw or not raw.strip():
        return None

    # Attempt 1: direct parse
    result = _try_parse(raw.strip())
    if result is not None:
        return result

    # Attempt 2: strip markdown fences
    stripped = _strip_markdown_fences(raw)
    result = _try_parse(stripped)
    if result is not None:
        return result

    # Attempt 3: extract first JSON object or array
    extracted = _extract_json_block(raw)
    if extracted:
        result = _try_parse(extracted)
        if result is not None:
            return result

    # Attempt 4: fix common issues and retry
    fixed = _fix_common_issues(stripped or raw)
    result = _try_parse(fixed)
    if result is not None:
        return result

    logger.warning("parse_llm_json: all parsing attempts failed for output: %.200s", raw)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_parse(text: str) -> Optional[dict | list]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json … ``` or ``` … ``` wrappers."""
    # Remove opening fence with optional language tag
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()


def _extract_json_block(text: str) -> Optional[str]:
    """
    Find the first { … } or [ … ] block by counting braces/brackets.
    Returns the substring or None.
    """
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start=start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _fix_common_issues(text: str) -> str:
    """Apply heuristic fixes for common LLM JSON formatting mistakes."""
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([\}\]])", r"\1", text)
    # Replace single quotes with double quotes (only at key/value positions)
    # This is risky but useful for simple cases
    # text = text.replace("'", '"')  # intentionally disabled — too aggressive
    return text


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------


def safe_get(data: dict, *keys: str, default: Any = None) -> Any:
    """Traverse nested dict using a sequence of keys; return default on miss."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is default:
            return default
    return current


def ensure_list(value: Any) -> list:
    """Wrap a non-list in a list; return [] for None/falsy."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
