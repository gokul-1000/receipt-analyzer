"""
services/image_service.py — Image preprocessing pipeline.

Applies a configurable sequence of image enhancement steps before OCR.
The goal is to maximise Tesseract accuracy without destroying useful
information.

All steps are optional and controlled by flags in config.py.
"""

from __future__ import annotations

import base64
import io
import logging
import math
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """Raised when image preprocessing fails unrecoverably."""
    pass


def load_image(source: bytes | str | io.IOBase) -> Image.Image:
    """
    Load a PIL Image from bytes, a file path, or a file-like object.

    Raises ImageProcessingError on failure.
    """
    try:
        if isinstance(source, bytes):
            img = Image.open(io.BytesIO(source))
        elif isinstance(source, str):
            img = Image.open(source)
        else:
            img = Image.open(source)
        img.load()  # Force decoding to catch corrupt files early
        return img
    except Exception as exc:
        raise ImageProcessingError(f"Failed to load image: {exc}") from exc


def preprocess_for_ocr(
    image: Image.Image,
    *,
    grayscale: bool = True,
    enhance_contrast: bool = True,
    sharpen: bool = True,
    denoise: bool = True,
    deskew: bool = True,
    min_width: int = 1000,
) -> Image.Image:
    """
    Apply preprocessing steps to improve OCR accuracy.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image (any mode).
    grayscale : bool
        Convert to grayscale first.
    enhance_contrast : bool
        Apply CLAHE-like contrast enhancement via Pillow.
    sharpen : bool
        Apply an unsharp mask to emphasise text edges.
    denoise : bool
        Apply a light median filter to reduce noise.
    deskew : bool
        Attempt to correct small rotation angles.
    min_width : int
        Upscale if the image is narrower than this (helps OCR on small scans).

    Returns
    -------
    PIL.Image.Image
        Preprocessed image.
    """
    try:
        img = image.copy()

        # --- Upscale if too small ---
        if img.width < min_width:
            scale = min_width / img.width
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.debug("Upscaled image to %dx%d", new_w, new_h)

        # --- Convert to RGB first (handles RGBA, P, CMYK, etc.) ---
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # --- Grayscale ---
        if grayscale and img.mode != "L":
            img = img.convert("L")

        # --- Denoise (before contrast to avoid amplifying noise) ---
        if denoise:
            img = img.filter(ImageFilter.MedianFilter(size=3))

        # --- Contrast enhancement ---
        if enhance_contrast:
            if img.mode == "L":
                img = ImageOps.autocontrast(img, cutoff=1)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)

        # --- Sharpening ---
        if sharpen:
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

        # --- Deskew ---
        if deskew:
            img = _deskew(img)

        return img

    except Exception as exc:
        logger.warning("Image preprocessing step failed: %s — returning original", exc)
        return image


def _deskew(image: Image.Image) -> Image.Image:
    """
    Estimate and correct skew angle using projection profile method.

    Only corrects angles within ±15°.  Returns original if deskew fails
    or angle is negligible (< 0.5°).
    """
    try:
        # Convert to numpy for angle detection
        img_array = np.array(image.convert("L") if image.mode != "L" else image)

        # Binarise
        threshold = 128
        binary = (img_array < threshold).astype(np.uint8)

        # Try angles from -15 to +15 degrees
        best_angle = 0.0
        best_score = -1.0
        angles = np.arange(-15, 15.5, 0.5)

        for angle in angles:
            rotated = _rotate_array(binary, angle)
            # Score = variance of row sums (text rows have high variance)
            row_sums = rotated.sum(axis=1).astype(float)
            score = float(np.var(row_sums))
            if score > best_score:
                best_score = score
                best_angle = angle

        if abs(best_angle) < 0.5:
            return image  # No significant skew

        logger.debug("Deskewing by %.1f degrees", best_angle)
        return image.rotate(best_angle, expand=True, fillcolor=255)

    except Exception as exc:
        logger.debug("Deskew failed: %s", exc)
        return image


def _rotate_array(array: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a binary numpy array by *angle* degrees."""
    pil_img = Image.fromarray((array * 255).astype(np.uint8))
    rotated = pil_img.rotate(angle, expand=True, fillcolor=0)
    return np.array(rotated) // 255


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert a PIL Image to bytes."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def bytes_to_pil(data: bytes) -> Image.Image:
    """Convert raw bytes to a PIL Image."""
    return Image.open(io.BytesIO(data))


def validate_image_file(data: bytes, filename: str) -> tuple[bool, str]:
    """
    Validate that the uploaded file is a readable image.

    Returns (is_valid, error_message).
    """
    from config import MAX_FILE_SIZE_MB

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(data) > max_bytes:
        return False, f"File exceeds maximum size of {MAX_FILE_SIZE_MB} MB."

    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # Lightweight integrity check
        return True, ""
    except Exception as exc:
        return False, f"File is corrupted or not a valid image: {exc}"


def prepare_image_for_vision(
    source: bytes | Image.Image,
    max_dimension: int = 2048,
    max_size_bytes: int = 10 * 1024 * 1024,
) -> tuple[str, str]:
    """
    Prepare an image for Groq Vision AI input:
    - Normalise format to RGB JPEG
    - Downscale if width/height exceed max_dimension while maintaining aspect ratio
    - Compress JPEG if byte size exceeds max_size_bytes
    - Return base64 string and MIME type ("image/jpeg")

    Raises ImageProcessingError on failure.
    """
    try:
        if isinstance(source, bytes):
            img = load_image(source)
        elif isinstance(source, Image.Image):
            img = source.copy()
        else:
            raise ImageProcessingError(f"Unsupported image input type: {type(source)}")

        # Convert to RGB if needed
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")

        # Downscale if necessary
        width, height = img.size
        if max(width, height) > max_dimension:
            scale = max_dimension / max(width, height)
            new_w = max(1, int(width * scale))
            new_h = max(1, int(height * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.debug("Resized image for vision: %dx%d -> %dx%d", width, height, new_w, new_h)

        # Save to buffer as JPEG
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        img_bytes = buf.getvalue()

        # If still too large, reduce quality progressively
        quality = 75
        while len(img_bytes) > max_size_bytes and quality >= 30:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            img_bytes = buf.getvalue()
            quality -= 15

        if len(img_bytes) > max_size_bytes:
            raise ImageProcessingError(
                f"Image size ({len(img_bytes) / 1024 / 1024:.1f} MB) exceeds maximum supported payload ({max_size_bytes / 1024 / 1024:.1f} MB)."
            )

        encoded = base64.b64encode(img_bytes).decode("utf-8")
        return encoded, "image/jpeg"

    except ImageProcessingError:
        raise
    except Exception as exc:
        raise ImageProcessingError(f"Failed to prepare image for vision: {exc}") from exc
