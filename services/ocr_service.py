"""
services/ocr_service.py — Tesseract OCR service.

Wraps pytesseract and provides:
- Robust cross-platform Tesseract executable discovery
- OCR from a PIL Image with configurable preprocessing
- Per-character OCR quality estimation
- Graceful error messages when Tesseract is not installed

Technology choice: Tesseract via pytesseract.
Rationale:
- Best-in-class open-source OCR with decades of development
- Works on Streamlit Cloud (installable via packages.txt)
- No GPU required
- Excellent accuracy on printed receipts after preprocessing
- Widely supported, stable API

Alternative considered: EasyOCR — better on handwritten text but
requires significantly more RAM and is slower on Streamlit Cloud free tier.

Tesseract discovery order (evaluated once at import time):
    1. TESSERACT_CMD environment variable (highest priority)
    2. Current PATH  (works on Linux / Streamlit Cloud / macOS)
    3. Common Windows installation directories
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from typing import Optional

from PIL import Image

from config import (
    OCR_MIN_CONFIDENCE,
    PREPROCESS_CONTRAST,
    PREPROCESS_DENOISE,
    PREPROCESS_DESKEW,
    PREPROCESS_GRAYSCALE,
    PREPROCESS_RESIZE_MIN_WIDTH,
    PREPROCESS_SHARPEN,
    TESSERACT_LANG,
    TESSERACT_OEM,
    TESSERACT_PSM,
)
from core.models import OCRResult
from services.image_service import preprocess_for_ocr
from utils.normalization import clean_ocr_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Windows candidate paths (checked in order when not found on PATH)
# ---------------------------------------------------------------------------

_WINDOWS_CANDIDATE_PATHS: list[str] = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{username}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    r"C:\Tesseract-OCR\tesseract.exe",
]


# ---------------------------------------------------------------------------
# Tesseract discovery
# ---------------------------------------------------------------------------


def find_tesseract() -> Optional[str]:
    """
    Locate the Tesseract executable using a prioritised search strategy.

    Search order
    ------------
    1. ``TESSERACT_CMD`` environment variable — if set and the path is
       executable, use it immediately.
    2. System PATH — ``shutil.which("tesseract")`` covers Linux, macOS,
       and Windows environments where the installer added Tesseract to PATH.
    3. Common Windows installation directories — checked only on Windows.

    Returns
    -------
    str or None
        Absolute path to the Tesseract executable, or ``None`` if not found.
    """
    # --- 1. Explicit environment variable ---
    env_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if env_cmd:
        if _is_executable(env_cmd):
            logger.debug("Tesseract found via TESSERACT_CMD: %s", env_cmd)
            return env_cmd
        else:
            logger.warning(
                "TESSERACT_CMD is set to '%s' but the file is not executable "
                "or does not exist. Falling back to PATH search.",
                env_cmd,
            )

    # --- 2. System PATH ---
    path_result = shutil.which("tesseract")
    if path_result:
        logger.debug("Tesseract found on PATH: %s", path_result)
        return path_result

    # --- 3. Windows common directories ---
    if platform.system() == "Windows":
        username = os.environ.get("USERNAME", "")
        for template in _WINDOWS_CANDIDATE_PATHS:
            candidate = template.replace("{username}", username)
            if _is_executable(candidate):
                logger.debug("Tesseract found at Windows default path: %s", candidate)
                return candidate

    return None


def _is_executable(path: str) -> bool:
    """Return True if *path* exists and is a regular file (or symlink to one)."""
    return bool(path) and os.path.isfile(path)


def _verify_tesseract(cmd: str) -> bool:
    """
    Run ``tesseract --version`` to confirm the binary actually works.

    Returns True on success, False on any failure.
    This is a lightweight sanity-check run once at startup.
    """
    try:
        result = subprocess.run(
            [cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version_line = (result.stdout or result.stderr or "").splitlines()
            version_info = version_line[0] if version_line else "unknown version"
            logger.info("Tesseract verified: %s", version_info)
            return True
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _build_not_found_message() -> str:
    """Return a human-readable installation hint for the current OS."""
    system = platform.system()
    if system == "Windows":
        return (
            "Tesseract OCR is not installed or could not be found.\n\n"
            "Install it from: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "After installing, either:\n"
            "  • Add it to your PATH, or\n"
            "  • Set the environment variable:  TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        )
    if system == "Darwin":
        return (
            "Tesseract OCR is not installed.\n"
            "Install via Homebrew:  brew install tesseract"
        )
    return (
        "Tesseract OCR is not installed.\n"
        "Install via apt:  sudo apt-get install tesseract-ocr\n"
        "Or add 'tesseract-ocr' to packages.txt for Streamlit Cloud."
    )


# ---------------------------------------------------------------------------
# Module-level initialisation (runs once on import)
# ---------------------------------------------------------------------------

try:
    import pytesseract as _pytesseract_module  # imported once; aliased below
    _PYTESSERACT_AVAILABLE = True
except ImportError:
    _pytesseract_module = None  # type: ignore[assignment]
    _PYTESSERACT_AVAILABLE = False
    logger.warning("pytesseract is not installed — OCR will not work.")

# Resolve and configure the Tesseract executable
_TESSERACT_CMD: Optional[str] = None
_TESSERACT_READY: bool = False

if _PYTESSERACT_AVAILABLE:
    _TESSERACT_CMD = find_tesseract()
    if _TESSERACT_CMD:
        # Tell pytesseract exactly where the binary is
        _pytesseract_module.pytesseract.tesseract_cmd = _TESSERACT_CMD
        _TESSERACT_READY = _verify_tesseract(_TESSERACT_CMD)
        if not _TESSERACT_READY:
            logger.warning(
                "Found Tesseract at '%s' but 'tesseract --version' failed. "
                "OCR may not work correctly.",
                _TESSERACT_CMD,
            )
    else:
        logger.warning(
            "Tesseract executable not found. %s", _build_not_found_message()
        )

# Public alias — the rest of the module uses this name
pytesseract = _pytesseract_module


class OCRError(Exception):
    """Raised when OCR fails unrecoverably."""
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_tesseract_available() -> bool:
    """Return True if both pytesseract and the Tesseract binary are ready."""
    return _PYTESSERACT_AVAILABLE and _TESSERACT_READY


def get_tesseract_cmd() -> Optional[str]:
    """Return the resolved Tesseract executable path, or None."""
    return _TESSERACT_CMD


def run_ocr(image: Image.Image) -> OCRResult:
    """
    Run Tesseract OCR on a PIL Image.

    The image is preprocessed internally (grayscale, contrast, deskew, etc.)
    before being passed to Tesseract.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image in any PIL-supported mode.

    Returns
    -------
    OCRResult
        Extracted text, quality score, and processing metadata.

    Raises
    ------
    OCRError
        If pytesseract is not installed or the Tesseract binary cannot be found.
    """
    if not _PYTESSERACT_AVAILABLE:
        raise OCRError(
            "pytesseract is not installed. "
            "Add 'pytesseract' to requirements.txt and re-deploy."
        )

    if not _TESSERACT_READY:
        raise OCRError(_build_not_found_message())

    # --- Preprocess ---
    try:
        processed = preprocess_for_ocr(
            image,
            grayscale=PREPROCESS_GRAYSCALE,
            enhance_contrast=PREPROCESS_CONTRAST,
            sharpen=PREPROCESS_SHARPEN,
            denoise=PREPROCESS_DENOISE,
            deskew=PREPROCESS_DESKEW,
            min_width=PREPROCESS_RESIZE_MIN_WIDTH,
        )
    except Exception as exc:
        logger.warning("Image preprocessing failed: %s — using original image", exc)
        processed = image

    # --- Run Tesseract ---
    tess_config = f"--oem {TESSERACT_OEM} --psm {TESSERACT_PSM}"
    try:
        raw_text: str = pytesseract.image_to_string(
            processed,
            lang=TESSERACT_LANG,
            config=tess_config,
        )
    except Exception as exc:
        raise OCRError(
            f"Tesseract OCR failed: {exc}\n\n"
            f"Tesseract path: {_TESSERACT_CMD}"
        ) from exc

    # --- Estimate quality ---
    quality = _estimate_quality(raw_text, processed)
    warnings: list[str] = []
    if quality < OCR_MIN_CONFIDENCE:
        warnings.append(
            f"OCR quality is low ({quality:.0%}). Results may be inaccurate. "
            "Try uploading a higher-resolution or cleaner image."
        )

    # --- Clean text ---
    cleaned = clean_ocr_text(raw_text)

    return OCRResult(
        text=cleaned,
        quality_score=quality,
        page_count=1,
        is_selectable_text=False,
        method_used="tesseract",
        warnings=warnings,
    )


def run_ocr_with_confidence_data(image: Image.Image) -> tuple[str, float]:
    """
    Run Tesseract and return (text, quality_score).

    Convenience wrapper for callers that only need the text and quality score.
    """
    result = run_ocr(image)
    return result.text, result.quality_score


# ---------------------------------------------------------------------------
# Quality estimation (internal)
# ---------------------------------------------------------------------------


def _estimate_quality(text: str, image: Image.Image) -> float:
    """
    Estimate OCR quality using Tesseract's per-word confidence data,
    falling back to a text-heuristic if that fails.
    """
    confidence_score = _tesseract_confidence(image)
    if confidence_score is not None:
        return confidence_score

    from utils.confidence import estimate_ocr_quality
    return estimate_ocr_quality(text)


def _tesseract_confidence(image: Image.Image) -> Optional[float]:
    """
    Use pytesseract.image_to_data to retrieve per-word confidence scores.

    Returns the mean confidence (normalised 0–1) for non-empty words,
    or None if the call fails.
    """
    if not _TESSERACT_READY:
        return None
    try:
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config=f"--oem {TESSERACT_OEM} --psm {TESSERACT_PSM}",
        )
        confidences = [
            int(c)
            for c, word in zip(data["conf"], data["text"])
            if str(c).isdigit() and int(c) >= 0 and word.strip()
        ]
        if not confidences:
            return None
        mean_conf = sum(confidences) / len(confidences)
        return round(mean_conf / 100.0, 4)   # normalise to [0, 1]
    except Exception as exc:
        logger.debug("Tesseract confidence data failed: %s", exc)
        return None
