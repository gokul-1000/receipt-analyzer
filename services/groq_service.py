"""
services/groq_service.py — Clean Groq API abstraction.

This service wraps all calls to the Groq API.  No other module should
import the Groq client directly.  This makes it easy to:
- swap models without changing business logic
- unit-test pipeline stages by mocking this service
- centralise error handling and retries
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from groq import Groq, APIStatusError, APITimeoutError, RateLimitError
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq package not installed — GroqService will raise on use.")


class GroqServiceError(Exception):
    """Raised when the Groq API call fails after all retries."""
    pass


class GroqService:
    """
    Thin abstraction over the Groq chat completion API.

    Parameters
    ----------
    api_key : str
        Groq API key.  Never pass None — use get_api_key() to retrieve it.
    model : str
        Primary model name.
    fallback_model : str
        Fallback model used when the primary model is rate-limited.
    max_retries : int
        Number of retry attempts on transient errors.
    retry_delay : float
        Seconds to wait between retries (exponential back-off applied).
    timeout : int
        Seconds before a request times out.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        fallback_model: str,
        vision_model: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: int = 60,
    ) -> None:
        if not GROQ_AVAILABLE:
            raise GroqServiceError(
                "groq package is not installed.  Run: pip install groq"
            )
        if not api_key:
            raise GroqServiceError(
                "GROQ_API_KEY is not set.  Add it to .env or Streamlit secrets."
            )
        self._client = Groq(api_key=api_key, timeout=timeout)
        self.model = model
        self.fallback_model = fallback_model
        self.vision_model = vision_model or get_groq_vision_model()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def extract_receipt(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """
        Stage 1: Send receipt OCR text to the LLM and return raw response text.

        Returns
        -------
        str
            Raw model output (expected to be JSON).

        Raises
        ------
        GroqServiceError
            If all retry attempts are exhausted.
        """
        return self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )

    def categorize_items(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> str:
        """
        Stage 2: Batch categorise extracted items.

        Returns
        -------
        str
            Raw model output (expected to be a JSON array).
        """
        return self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )

    def extract_receipt_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: str,
        mime_type: str = "image/jpeg",
        vision_model: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a receipt image directly to the Groq vision model with extraction instructions.

        Returns
        -------
        str
            Raw model output (expected to be JSON conforming to the Receipt schema).

        Raises
        ------
        GroqServiceError
            If all retry attempts are exhausted.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        },
                    },
                ],
            },
        ]

        target_model = vision_model or self.vision_model
        return self._chat_messages(
            messages=messages,
            models_to_try=[target_model],
            max_tokens=max_tokens,
            temperature=0.0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        """
        Execute a text chat completion with retry logic and model fallback.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._chat_messages(
            messages=messages,
            models_to_try=[self.model, self.fallback_model],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _chat_messages(
        self,
        messages: list[dict],
        models_to_try: list[str],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        """
        Execute chat completion with retry logic, timeout, and model fallback.
        """
        last_error: Optional[Exception] = None

        for model in models_to_try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.debug(
                        "Groq request: model=%s, attempt=%d, max_tokens=%d",
                        model, attempt, max_tokens,
                    )
                    response = self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    content = response.choices[0].message.content or ""
                    logger.debug("Groq response length: %d chars", len(content))
                    return content

                except RateLimitError as exc:
                    logger.warning(
                        "Rate limit hit (model=%s, attempt=%d): %s",
                        model, attempt, exc,
                    )
                    last_error = exc
                    # Try fallback model immediately on rate limit
                    break

                except APITimeoutError as exc:
                    logger.warning(
                        "Timeout (model=%s, attempt=%d): %s",
                        model, attempt, exc,
                    )
                    last_error = exc
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * attempt)

                except APIStatusError as exc:
                    logger.error(
                        "API status error (model=%s, attempt=%d, status=%s): %s",
                        model, attempt, exc.status_code, exc,
                    )
                    last_error = exc
                    if exc.status_code in (429, 500, 502, 503, 504):
                        if attempt < self.max_retries:
                            time.sleep(self.retry_delay * attempt)
                    else:
                        # Non-retriable error (e.g. 400 bad request)
                        raise GroqServiceError(
                            f"Groq API error {exc.status_code}: {exc.message}"
                        ) from exc

                except Exception as exc:
                    logger.error(
                        "Unexpected Groq error (model=%s, attempt=%d): %s",
                        model, attempt, exc,
                    )
                    last_error = exc
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * attempt)

        raise GroqServiceError(
            f"Groq API call failed after all retries. Last error: {last_error}"
        ) from last_error


# ---------------------------------------------------------------------------
# Factory / key retrieval
# ---------------------------------------------------------------------------


def get_api_key() -> str:
    """
    Retrieve the Groq API key from Streamlit secrets or environment variables.

    Priority:
    1. Streamlit secrets (st.secrets["GROQ_API_KEY"]) — used in Streamlit Cloud.
    2. Environment variable GROQ_API_KEY — used locally with .env.

    Never returns the key to the frontend / browser.
    """
    # Try Streamlit secrets first
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    # Fall back to environment variable
    key = os.getenv("GROQ_API_KEY", "")
    if key:
        return key

    return ""


def get_groq_model() -> str:
    """
    Retrieve the configured Groq model name.

    Priority:
    1. Streamlit secrets (st.secrets["GROQ_MODEL"])
    2. Environment variable GROQ_MODEL
    3. config.GROQ_MODEL (default: openai/gpt-oss-120b)
    """
    try:
        import streamlit as st
        secret_model = st.secrets.get("GROQ_MODEL", "")
        if secret_model:
            return secret_model
    except Exception:
        pass

    env_model = os.getenv("GROQ_MODEL", "").strip()
    if env_model:
        return env_model

    import config
    return getattr(config, "GROQ_MODEL", "openai/gpt-oss-120b")


def get_groq_vision_model() -> str:
    """
    Retrieve the configured Groq vision model name.

    Priority:
    1. Streamlit secrets (st.secrets["GROQ_VISION_MODEL"])
    2. Environment variable GROQ_VISION_MODEL
    3. config.GROQ_VISION_MODEL (default: qwen/qwen3.6-27b)
    """
    try:
        import streamlit as st
        secret_model = st.secrets.get("GROQ_VISION_MODEL", "")
        if secret_model:
            return secret_model
    except Exception:
        pass

    env_model = os.getenv("GROQ_VISION_MODEL", "").strip()
    if env_model:
        return env_model

    import config
    return getattr(config, "GROQ_VISION_MODEL", "qwen/qwen3.6-27b")


def create_groq_service(
    model: Optional[str] = None,
    fallback_model: Optional[str] = None,
    vision_model: Optional[str] = None,
) -> GroqService:
    """
    Convenience factory that reads config and creates a ready-to-use GroqService instance.
    """
    from config import (
        GROQ_FALLBACK_MODEL,
        GROQ_MAX_RETRIES,
        GROQ_RETRY_DELAY_SECONDS,
        GROQ_TIMEOUT_SECONDS,
    )

    api_key = get_api_key()
    selected_model = model or get_groq_model()
    selected_fallback = (
        fallback_model
        or os.getenv("GROQ_FALLBACK_MODEL", "").strip()
        or GROQ_FALLBACK_MODEL
    )
    selected_vision = vision_model or get_groq_vision_model()

    return GroqService(
        api_key=api_key,
        model=selected_model,
        fallback_model=selected_fallback,
        vision_model=selected_vision,
        max_retries=GROQ_MAX_RETRIES,
        retry_delay=GROQ_RETRY_DELAY_SECONDS,
        timeout=GROQ_TIMEOUT_SECONDS,
    )
