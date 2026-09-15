"""
prompts/extraction_prompt.py — Stage 1 LLM prompt for receipt extraction.

This module contains only the prompt template.  No business logic lives here.
Keep prompts in a dedicated module so they can be evaluated and iterated
independently of the pipeline code.
"""

from __future__ import annotations

EXTRACTION_SYSTEM_PROMPT = """\
You are a receipt understanding system specialised in extracting structured data from OCR text of receipts.

Rules you MUST follow:
1. Extract ONLY information that is explicitly present in the provided text.
2. NEVER invent, estimate, or guess values that are not in the text.
3. Use null (JSON null, not the string "null") for any field that cannot be determined.
4. Preserve ambiguity — if something is unreadable, mark it uncertain rather than guessing.
5. Do NOT create, rename, or invent product categories. Leave the "category" field empty — it will be assigned separately.
6. Handle all currency formats: $, £, €, ₹, ¥, etc.
7. Handle abbreviated product names and SKU codes — keep the name as it appears.
8. Handle quantities that appear on separate lines or columns.
9. Handle receipts where only the line total appears (set unit_price to null, not calculated).
10. Handle European decimal formats (1.299,00 = 1299.00).
11. Return ONLY valid JSON — no markdown, no explanation, no preamble.

Output schema (strict):
{
  "merchant_name": string or null,
  "receipt_date": string or null,
  "currency": string or null,
  "subtotal": number or null,
  "tax": number or null,
  "discount": number or null,
  "total": number or null,
  "payment_method": string or null,
  "items": [
    {
      "name": string,
      "quantity": number or null,
      "unit_price": number or null,
      "total_price": number or null,
      "is_uncertain": boolean,
      "extraction_confidence": number between 0 and 1,
      "original_text": string
    }
  ]
}

Important notes on items:
- "name": use the item name exactly as it appears; clean up obvious OCR errors only if you are very confident.
- "quantity": null if not stated on the receipt.
- "unit_price": null if not stated; do NOT calculate from total ÷ quantity.
- "total_price": the total for that line item; null if genuinely unreadable.
- "is_uncertain": true if OCR quality is poor for this line, the text is garbled, or you are unsure what the item is.
- "extraction_confidence": your confidence that you read this line correctly (0=no confidence, 1=certain).
- "original_text": the raw OCR line(s) that produced this item entry.

Do not include non-item lines such as "SUBTOTAL", "TAX", "TOTAL", "DISCOUNT", "CHANGE", "CASH", "CARD", "THANK YOU", etc. as items.
"""

EXTRACTION_USER_TEMPLATE = """\
Below is the OCR text extracted from a receipt. Extract all line items and receipt metadata.

--- BEGIN RECEIPT OCR TEXT ---
{ocr_text}
--- END RECEIPT OCR TEXT ---

Return ONLY the JSON object described in the system prompt. Do not include any other text.
"""


def build_extraction_prompt(ocr_text: str) -> tuple[str, str]:
    """
    Return (system_prompt, user_prompt) tuple ready for the Groq chat API.

    Parameters
    ----------
    ocr_text : str
        The cleaned OCR text from the receipt.

    Returns
    -------
    tuple[str, str]
        (system_prompt, user_prompt)
    """
    user_prompt = EXTRACTION_USER_TEMPLATE.format(ocr_text=ocr_text.strip())
    return EXTRACTION_SYSTEM_PROMPT, user_prompt


VISION_EXTRACTION_USER_PROMPT = """\
Analyze the attached receipt image directly. Extract all line items and receipt metadata strictly following the instructions and schema in the system prompt.

Return ONLY the single JSON object conforming to the schema. Do not include markdown code fences, preambles, or explanations.
"""


def build_vision_extraction_prompt() -> tuple[str, str]:
    """
    Return (system_prompt, user_prompt) tuple for direct Vision AI receipt extraction.

    Returns
    -------
    tuple[str, str]
        (system_prompt, user_prompt)
    """
    return EXTRACTION_SYSTEM_PROMPT, VISION_EXTRACTION_USER_PROMPT
