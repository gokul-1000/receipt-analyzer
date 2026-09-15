"""
utils/normalization.py — Normalise price strings, dates, currencies,
and quantities extracted from OCR or LLM output.

All functions are pure (no side-effects) and do NOT call the LLM.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Currency symbol → code mapping
# ---------------------------------------------------------------------------

CURRENCY_SYMBOL_MAP: dict[str, str] = {
    "$": "USD",
    "£": "GBP",
    "€": "EUR",
    "₹": "INR",
    "¥": "JPY",
    "₩": "KRW",
    "₫": "VND",
    "฿": "THB",
    "₺": "TRY",
    "R": "ZAR",
    "kr": "SEK",
    "Fr": "CHF",
    "A$": "AUD",
    "C$": "CAD",
    "NZ$": "NZD",
    "S$": "SGD",
    "HK$": "HKD",
    "₦": "NGN",
    "₱": "PHP",
    "RM": "MYR",
    "Rp": "IDR",
    "د.إ": "AED",
    "﷼": "SAR",
}

_REVERSE_SYMBOL_MAP: dict[str, str] = {v.lower(): v for v in CURRENCY_SYMBOL_MAP.values()}


def normalize_currency(raw: Optional[str]) -> Optional[str]:
    """
    Return a cleaned currency string (symbol or 3-letter code).
    Returns None if *raw* is empty/None.
    """
    if not raw:
        return None
    raw = raw.strip()
    # Already looks like ISO code
    if re.match(r"^[A-Z]{3}$", raw):
        return raw
    # Symbol in map
    if raw in CURRENCY_SYMBOL_MAP:
        return raw
    # Case-insensitive ISO lookup
    upper = raw.upper()
    if len(upper) == 3 and upper.isalpha():
        return upper
    return raw or None


def normalize_price(raw: Optional[str | float | int]) -> Optional[float]:
    """
    Parse a price value from various formats:

    Examples accepted:
        "$1,299.00", "1.299,00", "1299", "₹ 1,299", "1.2k"

    Returns None if parsing fails.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw >= 0 else None

    text = str(raw).strip()
    # Remove currency symbols and whitespace
    text = re.sub(r"[^\d.,k]", "", text, flags=re.IGNORECASE)

    if not text:
        return None

    # Handle "k" suffix (e.g. "1.2k" → 1200)
    if text.lower().endswith("k"):
        try:
            return float(text[:-1]) * 1000
        except ValueError:
            return None

    # Detect decimal separator:
    # European format "1.299,00" → comma is decimal, dot is thousands
    # US/UK format    "1,299.00" → dot is decimal, comma is thousands
    # Ambiguous       "1,299"    → single comma, 3 digits after → thousands separator
    dot_idx = text.rfind(".")
    comma_idx = text.rfind(",")

    if dot_idx > comma_idx:
        # Dot is last separator → decimal point (e.g. 1,299.00)
        text = text.replace(",", "")
    elif comma_idx > dot_idx:
        # Check if this is an ambiguous thousands separator:
        # "1,299" — single comma, exactly 3 digits after it, no dot
        after_comma = text[comma_idx + 1:]
        if dot_idx == -1 and len(after_comma) == 3 and after_comma.isdigit():
            # Treat as thousands separator (US/Indian format: 1,299 → 1299)
            text = text.replace(",", "")
        else:
            # Comma is last separator → decimal point (e.g. 1.299,00)
            text = text.replace(".", "").replace(",", ".")
    else:
        # Only one type or none
        text = text.replace(",", "")

    try:
        value = float(text)
        return value if value >= 0 else None
    except ValueError:
        return None


def normalize_quantity(raw: Optional[str | float | int]) -> Optional[float]:
    """Parse a quantity value.  Returns None if invalid or zero."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None
    text = str(raw).strip()
    # Remove whitespace and common non-numeric prefix/suffix
    text = re.sub(r"[^\d.]", "", text)
    try:
        q = float(text)
        return q if q > 0 else None
    except ValueError:
        return None


def normalize_date(raw: Optional[str]) -> Optional[str]:
    """
    Best-effort date normalisation.  Returns the input unchanged if
    it cannot be parsed (we prefer ambiguity over hallucination).

    Attempts common formats:  DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, etc.
    """
    if not raw:
        return None

    raw = raw.strip()

    # Already ISO (YYYY-MM-DD)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw

    # Try common formats
    import datetime

    formats = [
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d.%m.%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(raw, fmt)
            return dt.date().isoformat()
        except ValueError:
            continue

    # Return original if nothing matches
    return raw


def clean_ocr_text(text: str) -> str:
    """
    Light cleanup of raw OCR output:
    - Collapse multiple blank lines
    - Strip trailing whitespace per line
    - Remove non-printable characters

    Does NOT remove any potentially useful tokens.
    """
    # Replace non-printable chars (but preserve newlines)
    text = re.sub(r"[^\x20-\x7E\n\r\t]", " ", text)
    # Strip trailing spaces per line
    lines = [line.rstrip() for line in text.splitlines()]
    # Collapse 3+ consecutive blank lines to 2
    cleaned: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def extract_currency_from_text(text: str) -> Optional[str]:
    """
    Scan OCR text for the most likely currency symbol.
    Returns the first symbol found, or None.
    """
    # Check multi-char symbols first to avoid partial matches
    multi = sorted(CURRENCY_SYMBOL_MAP.keys(), key=len, reverse=True)
    for sym in multi:
        if sym in text:
            return sym
    return None
