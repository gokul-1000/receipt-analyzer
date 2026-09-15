"""
tests/test_model_config.py — Tests for Groq model configuration and discovery.

Verifies:
- Default model is openai/gpt-oss-120b
- Environment variable GROQ_MODEL overrides default model
- Streamlit secrets GROQ_MODEL overrides default model
- Explicit model parameter in create_groq_service overrides default
- Configurable model attribute on GroqService instance
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from services.groq_service import (
    GroqService,
    create_groq_service,
    get_groq_model,
)


class TestGroqModelConfiguration:
    def test_default_model_is_gpt_oss_120b(self, monkeypatch):
        """Default model should be openai/gpt-oss-120b when no env or secret is set."""
        monkeypatch.delenv("GROQ_MODEL", raising=False)
        with patch.dict("sys.modules", {"streamlit": None}):
            # Ensure config.GROQ_MODEL default is openai/gpt-oss-120b
            assert config.get_default_groq_model() == "openai/gpt-oss-120b"
            assert get_groq_model() == "openai/gpt-oss-120b"

    def test_env_var_overrides_model(self, monkeypatch):
        """Setting GROQ_MODEL environment variable should override default."""
        custom_model = "custom-model-from-env"
        monkeypatch.setenv("GROQ_MODEL", custom_model)

        assert config.get_default_groq_model() == custom_model
        assert get_groq_model() == custom_model

    def test_streamlit_secrets_override_model(self, monkeypatch):
        """Streamlit secrets should take precedence for GROQ_MODEL."""
        monkeypatch.delenv("GROQ_MODEL", raising=False)
        mock_st = MagicMock()
        mock_st.secrets = {"GROQ_MODEL": "model-from-secrets"}

        with patch.dict("sys.modules", {"streamlit": mock_st}):
            assert get_groq_model() == "model-from-secrets"

    def test_create_groq_service_uses_configured_model(self, monkeypatch):
        """create_groq_service should use the configured model."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")
        monkeypatch.delenv("GROQ_MODEL", raising=False)

        with patch("services.groq_service.Groq", return_value=MagicMock()):
            service = create_groq_service()
            assert service.model == "openai/gpt-oss-120b"

    def test_create_groq_service_explicit_override(self, monkeypatch):
        """Explicit model argument should take precedence over configured model."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key")
        monkeypatch.setenv("GROQ_MODEL", "model-from-env")

        with patch("services.groq_service.Groq", return_value=MagicMock()):
            service = create_groq_service(model="explicit-model-123")
            assert service.model == "explicit-model-123"

    def test_groq_service_model_attribute(self):
        """GroqService instance should store and use the configured model."""
        with patch("services.groq_service.Groq", return_value=MagicMock()):
            service = GroqService(
                api_key="gsk_test_key",
                model="openai/gpt-oss-120b",
                fallback_model="llama-3.1-8b-instant",
            )
            assert service.model == "openai/gpt-oss-120b"
            assert service.fallback_model == "llama-3.1-8b-instant"
