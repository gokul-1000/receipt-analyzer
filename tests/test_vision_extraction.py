"""
tests/test_vision_extraction.py — Focused tests for Vision AI extraction mode.

Verifies:
- Vision prompt generation and schema consistency
- Image preprocessing, format normalisation, and base64 encoding
- Handling of oversized and invalid images
- Groq multimodal payload construction using qwen/qwen3.6-27b
- End-to-end pipeline execution in Vision AI mode
- Graceful fallback to OCR + LLM upon vision errors or invalid JSON
"""

from __future__ import annotations

import base64
import io
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from core.models import Receipt, ValidationStatus
from core.pipeline import process_receipt
from prompts.extraction_prompt import (
    EXTRACTION_SYSTEM_PROMPT,
    build_vision_extraction_prompt,
)
from services.groq_service import (
    GroqService,
    GroqServiceError,
    create_groq_service,
    get_groq_vision_model,
)
from services.image_service import (
    ImageProcessingError,
    prepare_image_for_vision,
)


# ---------------------------------------------------------------------------
# Helpers & Sample Data
# ---------------------------------------------------------------------------

SAMPLE_VISION_JSON = """{
  "merchant_name": "Target Store #1042",
  "receipt_date": "2026-05-12",
  "currency": "$",
  "subtotal": 42.00,
  "tax": 3.50,
  "discount": null,
  "total": 45.50,
  "payment_method": "MASTERCARD",
  "items": [
    {
      "name": "Wireless Charging Pad",
      "quantity": 1,
      "unit_price": 25.00,
      "total_price": 25.00,
      "is_uncertain": false,
      "extraction_confidence": 0.96,
      "original_text": "Wireless Charging Pad $25.00"
    },
    {
      "name": "Organic Almond Milk",
      "quantity": 2,
      "unit_price": 4.50,
      "total_price": 9.00,
      "is_uncertain": false,
      "extraction_confidence": 0.98,
      "original_text": "2x Organic Almond Milk @ 4.50 = 9.00"
    },
    {
      "name": "Ceramic Mug",
      "quantity": 1,
      "unit_price": 8.00,
      "total_price": 8.00,
      "is_uncertain": false,
      "extraction_confidence": 0.92,
      "original_text": "Ceramic Mug $8.00"
    }
  ]
}"""

SAMPLE_CATEGORIZATION_JSON = """[
  {"item_index": 0, "category": "Electronics & Accessories", "confidence": 0.95, "reasoning": "Phone accessory"},
  {"item_index": 1, "category": "Food & Beverages", "confidence": 0.99, "reasoning": "Beverage / dairy alternative"},
  {"item_index": 2, "category": "Home & Kitchen", "confidence": 0.93, "reasoning": "Drinkware"}
]"""


def _create_sample_image_bytes(width: int = 200, height: int = 200, color: str = "white") -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Prompt & Model Config Tests
# ---------------------------------------------------------------------------

class TestVisionPromptAndConfig:
    def test_build_vision_extraction_prompt(self):
        """Vision prompt must use identical system prompt and rules as text extraction."""
        sys_prompt, user_prompt = build_vision_extraction_prompt()
        assert sys_prompt == EXTRACTION_SYSTEM_PROMPT
        assert "receipt" in user_prompt.lower()
        assert "schema" in user_prompt.lower()

    def test_default_vision_model_is_qwen36(self, monkeypatch):
        """Default vision model should be verified active model qwen/qwen3.6-27b."""
        monkeypatch.delenv("GROQ_VISION_MODEL", raising=False)
        with patch.dict("sys.modules", {"streamlit": None}):
            assert config.get_default_groq_vision_model() == "qwen/qwen3.6-27b"
            assert get_groq_vision_model() == "qwen/qwen3.6-27b"

    def test_env_var_overrides_vision_model(self, monkeypatch):
        """GROQ_VISION_MODEL env var should override the default vision model."""
        custom = "qwen/qwen-custom-vision"
        monkeypatch.setenv("GROQ_VISION_MODEL", custom)
        assert get_groq_vision_model() == custom


# ---------------------------------------------------------------------------
# Image Preparation for Vision Tests
# ---------------------------------------------------------------------------

class TestPrepareImageForVision:
    def test_valid_image_encodes_to_base64_jpeg(self):
        """PNG image should be converted to RGB JPEG base64."""
        img_bytes = _create_sample_image_bytes(100, 100)
        b64_str, mime = prepare_image_for_vision(img_bytes)

        assert mime == "image/jpeg"
        assert isinstance(b64_str, str)
        # Verify base64 decodes back to a valid JPEG
        decoded = base64.b64decode(b64_str)
        pil_img = Image.open(io.BytesIO(decoded))
        assert pil_img.format == "JPEG"

    def test_oversized_image_downscaled(self):
        """Images larger than max_dimension should be scaled down while preserving aspect ratio."""
        large_bytes = _create_sample_image_bytes(3000, 1500)
        b64_str, mime = prepare_image_for_vision(large_bytes, max_dimension=1000)

        decoded = base64.b64decode(b64_str)
        pil_img = Image.open(io.BytesIO(decoded))
        w, h = pil_img.size
        assert max(w, h) <= 1000
        assert round(w / h, 1) == 2.0

    def test_invalid_image_raises_error(self):
        """Corrupt or non-image bytes should raise ImageProcessingError."""
        corrupt_bytes = b"not an image file at all"
        with pytest.raises(ImageProcessingError):
            prepare_image_for_vision(corrupt_bytes)


# ---------------------------------------------------------------------------
# GroqService Multimodal Request Payload
# ---------------------------------------------------------------------------

class TestGroqServiceVisionCall:
    def test_extract_receipt_vision_payload_structure(self):
        """Verify chat.completions.create is called with the OpenAI-compatible multimodal format."""
        mock_groq_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=SAMPLE_VISION_JSON))]
        mock_groq_client.chat.completions.create.return_value = mock_response

        with patch("services.groq_service.Groq", return_value=mock_groq_client):
            service = GroqService(
                api_key="gsk_mock_test_key",
                model="openai/gpt-oss-120b",
                fallback_model="llama-3.1-8b-instant",
                vision_model="qwen/qwen3.6-27b",
            )

            result = service.extract_receipt_vision(
                system_prompt="Test System",
                user_prompt="Test User",
                image_base64="dGVzdA==",
                mime_type="image/jpeg",
            )

            assert result == SAMPLE_VISION_JSON

            # Check arguments sent to Groq client
            args, kwargs = mock_groq_client.chat.completions.create.call_args
            assert kwargs["model"] == "qwen/qwen3.6-27b"
            messages = kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"

            user_content = messages[1]["content"]
            assert isinstance(user_content, list)
            assert user_content[0]["type"] == "text"
            assert user_content[1]["type"] == "image_url"
            assert user_content[1]["image_url"]["url"] == "data:image/jpeg;base64,dGVzdA=="


# ---------------------------------------------------------------------------
# End-to-End Pipeline Vision Mode Tests
# ---------------------------------------------------------------------------

class TestPipelineVisionMode:
    def test_process_receipt_vision_mode_success(self):
        """Pipeline in vision mode parses receipt, validates arithmetic, and categorises."""
        img_bytes = _create_sample_image_bytes(400, 600)

        mock_groq = MagicMock(spec=GroqService)
        mock_groq.vision_model = "qwen/qwen3.6-27b"
        mock_groq.extract_receipt_vision.return_value = SAMPLE_VISION_JSON
        mock_groq.categorize_items.return_value = SAMPLE_CATEGORIZATION_JSON

        receipt = process_receipt(
            file_bytes=img_bytes,
            filename="target_receipt.png",
            groq=mock_groq,
            extraction_mode="vision",
        )

        assert isinstance(receipt, Receipt)
        assert receipt.merchant_name == "Target Store #1042"
        assert receipt.total == 45.50
        assert receipt.item_count == 3
        assert receipt.items[0].name == "Wireless Charging Pad"
        assert receipt.items[0].category == "Electronics & Accessories"
        assert receipt.items[1].name == "Organic Almond Milk"
        assert receipt.items[1].category == "Food & Beverages"
        assert receipt.validation.status in (ValidationStatus.VALIDATED, ValidationStatus.PARTIALLY_VALIDATED)
        assert "[Direct Vision AI Extraction" in receipt.ocr_text
        mock_groq.extract_receipt_vision.assert_called_once()

    def test_process_receipt_vision_mode_fallback_on_api_error(self):
        """If Vision AI fails with API error, pipeline gracefully falls back to OCR + LLM."""
        img_bytes = _create_sample_image_bytes(400, 600)

        mock_groq = MagicMock(spec=GroqService)
        mock_groq.vision_model = "qwen/qwen3.6-27b"
        # Vision fails
        mock_groq.extract_receipt_vision.side_effect = GroqServiceError("Vision rate limit exceeded")
        # Text extraction succeeds
        mock_groq.extract_receipt.return_value = SAMPLE_VISION_JSON
        mock_groq.categorize_items.return_value = SAMPLE_CATEGORIZATION_JSON

        with patch("core.pipeline._extract_text") as mock_ocr:
            mock_ocr.return_value = MagicMock(text="Sample OCR Text", quality_score=0.85, page_count=1)
            receipt = process_receipt(
                file_bytes=img_bytes,
                filename="target_receipt.png",
                groq=mock_groq,
                extraction_mode="vision",
            )

        assert isinstance(receipt, Receipt)
        assert receipt.merchant_name == "Target Store #1042"
        assert any("fell back to OCR + LLM" in note for note in receipt.validation.notes)

    def test_process_receipt_vision_mode_fallback_on_malformed_json(self):
        """If Vision AI returns invalid JSON, pipeline falls back to OCR + LLM."""
        img_bytes = _create_sample_image_bytes(400, 600)

        mock_groq = MagicMock(spec=GroqService)
        mock_groq.vision_model = "qwen/qwen3.6-27b"
        # Vision returns malformed non-JSON
        mock_groq.extract_receipt_vision.return_value = "Not valid json response at all"
        mock_groq.extract_receipt.return_value = SAMPLE_VISION_JSON
        mock_groq.categorize_items.return_value = SAMPLE_CATEGORIZATION_JSON

        with patch("core.pipeline._extract_text") as mock_ocr:
            mock_ocr.return_value = MagicMock(text="Sample OCR Text", quality_score=0.85, page_count=1)
            receipt = process_receipt(
                file_bytes=img_bytes,
                filename="receipt.jpg",
                groq=mock_groq,
                extraction_mode="vision",
            )

        assert isinstance(receipt, Receipt)
        assert receipt.item_count == 3
