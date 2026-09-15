"""
config.py — Central configuration for Receipt Intelligence.

All tuneable parameters, model names, thresholds, and category lists
live here so they can be changed without touching business logic.
"""

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Groq model configuration
# ---------------------------------------------------------------------------

def get_default_groq_model() -> str:
    """Resolve default Groq model from Streamlit secrets, environment, or default."""
    try:
        import streamlit as st
        secret_model = st.secrets.get("GROQ_MODEL", "")
        if secret_model:
            return secret_model
    except Exception:
        pass
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip() or "openai/gpt-oss-120b"


GROQ_MODEL: str = get_default_groq_model()
"""Primary Groq model for extraction and categorisation.
Default: openai/gpt-oss-120b (llama-3.3-70b-versatile is deprecated).
Override via Streamlit secrets (GROQ_MODEL) or environment variable."""

def get_default_groq_vision_model() -> str:
    """Resolve default Groq vision model from Streamlit secrets, environment, or default."""
    try:
        import streamlit as st
        secret_model = st.secrets.get("GROQ_VISION_MODEL", "")
        if secret_model:
            return secret_model
    except Exception:
        pass
    return os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b").strip() or "qwen/qwen3.6-27b"


GROQ_VISION_MODEL: str = get_default_groq_vision_model()
"""Primary Groq vision model for direct multimodal receipt extraction.
Default: qwen/qwen3.6-27b.
Override via Streamlit secrets (GROQ_VISION_MODEL) or environment variable."""

GROQ_FALLBACK_MODEL: str = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant").strip() or "llama-3.1-8b-instant"
"""Lightweight fallback model used when the primary model is rate-limited."""

GROQ_MAX_TOKENS_EXTRACTION: int = 4096
GROQ_MAX_TOKENS_CATEGORIZATION: int = 2048
GROQ_TEMPERATURE: float = 0.0          # Deterministic output for structured extraction
GROQ_TIMEOUT_SECONDS: int = 60
GROQ_MAX_RETRIES: int = 3
GROQ_RETRY_DELAY_SECONDS: float = 2.0  # Back-off between retries

# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_MB: int = 20
SUPPORTED_IMAGE_TYPES: tuple = ("jpg", "jpeg", "png", "webp")
SUPPORTED_PDF_TYPES: tuple = ("pdf",)
SUPPORTED_TYPES: tuple = SUPPORTED_IMAGE_TYPES + SUPPORTED_PDF_TYPES

PDF_MAX_PAGES: int = 10   # Safety cap for multi-page PDFs
PDF_DPI: int = 200        # Resolution for rasterising scanned PDFs

# ---------------------------------------------------------------------------
# OCR configuration
# ---------------------------------------------------------------------------

TESSERACT_PSM: int = 6              # Assume a single uniform block of text
TESSERACT_OEM: int = 3              # Default, based on what is available
TESSERACT_LANG: str = "eng"
OCR_MIN_CONFIDENCE: float = 0.30    # Below this → treat as low-quality OCR

# Image preprocessing flags (all enabled by default)
PREPROCESS_GRAYSCALE: bool = True
PREPROCESS_CONTRAST: bool = True
PREPROCESS_SHARPEN: bool = True
PREPROCESS_DENOISE: bool = True
PREPROCESS_DESKEW: bool = True
PREPROCESS_RESIZE_MIN_WIDTH: int = 1000   # Upscale narrow images for better OCR

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------

LINE_ITEM_TOLERANCE: float = 0.02     # 2 % rounding tolerance for qty × price
SUBTOTAL_TOLERANCE: float = 0.05      # 5 % tolerance for Σ(lines) ≈ subtotal
TOTAL_TOLERANCE: float = 0.05         # 5 % tolerance for subtotal + tax ≈ total

# ---------------------------------------------------------------------------
# Confidence scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------

CONFIDENCE_WEIGHT_OCR_QUALITY: float = 0.20
CONFIDENCE_WEIGHT_STRUCTURAL: float = 0.25
CONFIDENCE_WEIGHT_ARITHMETIC: float = 0.30
CONFIDENCE_WEIGHT_CATEGORY: float = 0.25

# Thresholds for confidence tiers
CONFIDENCE_HIGH: float = 0.85    # 🟢 High
CONFIDENCE_MEDIUM: float = 0.60  # 🟡 Medium
# Below CONFIDENCE_MEDIUM          # 🔴 Low

# ---------------------------------------------------------------------------
# Category taxonomy
# ---------------------------------------------------------------------------

CATEGORY_LIST: list[str] = [
    "Food & Beverages",
    "Clothing & Footwear",
    "Electronics & Accessories",
    "Health & Personal Care",
    "Beauty & Cosmetics",
    "Home & Kitchen",
    "Grocery",
    "Household Supplies",
    "Furniture",
    "Sports & Fitness",
    "Books & Stationery",
    "Toys & Games",
    "Automotive",
    "Pet Supplies",
    "Baby Products",
    "Travel & Luggage",
    "Jewelry & Accessories",
    "Office Supplies",
    "Hardware & Tools",
    "Services",
    "Other / Uncategorized",
]

# Human-readable descriptions used in the categorisation prompt
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "Food & Beverages": (
        "Prepared food, restaurant meals, drinks, coffee, alcohol, snacks, "
        "confectionery, protein shakes/bars intended as food."
    ),
    "Clothing & Footwear": (
        "Shirts, trousers, dresses, jackets, shoes, boots, sandals, socks, "
        "underwear, hats, belts, scarves."
    ),
    "Electronics & Accessories": (
        "Phones, laptops, tablets, chargers, cables, earphones, speakers, "
        "batteries, cameras, smart watches, USB accessories, phone cases."
    ),
    "Health & Personal Care": (
        "Medicines, vitamins, supplements, first-aid, medical devices, "
        "toothbrush, toothpaste, deodorant, shaving products, hygiene products."
    ),
    "Beauty & Cosmetics": (
        "Makeup, foundation, lipstick, mascara, eyeliner, perfume, skincare "
        "creams, moisturiser, sunscreen, nail polish, hair colour."
    ),
    "Home & Kitchen": (
        "Cookware, cutlery, kitchen appliances, plates, glasses, storage "
        "containers, bedding, curtains, rugs, lamps, small furniture pieces."
    ),
    "Grocery": (
        "Raw or packaged food items bought from a supermarket: vegetables, "
        "fruit, meat, dairy, eggs, bread, pasta, rice, canned goods, spices, "
        "oils, condiments."
    ),
    "Household Supplies": (
        "Cleaning products, detergents, bleach, mops, brooms, trash bags, "
        "paper towels, toilet paper, air fresheners, pest control."
    ),
    "Furniture": (
        "Sofas, chairs, tables, beds, wardrobes, shelves, desks, cabinets."
    ),
    "Sports & Fitness": (
        "Gym equipment, yoga mats, running shoes (if sport-specific context), "
        "sports clothing, bicycles, helmets, fitness trackers, protein powder "
        "marketed as sports supplement."
    ),
    "Books & Stationery": (
        "Books, magazines, notebooks, pens, pencils, highlighters, paper, "
        "folders, sticky notes, rulers."
    ),
    "Toys & Games": (
        "Children's toys, board games, video games, puzzles, action figures, "
        "dolls, remote-control cars, LEGO."
    ),
    "Automotive": (
        "Car parts, engine oil, tyres, car accessories, car cleaning products, "
        "fuel (petrol/diesel), air filters, wipers."
    ),
    "Pet Supplies": (
        "Pet food, pet toys, pet bedding, grooming products, veterinary "
        "supplies, leashes, cages."
    ),
    "Baby Products": (
        "Nappies/diapers, baby food, baby wipes, baby clothing, prams, "
        "feeding bottles, baby monitors."
    ),
    "Travel & Luggage": (
        "Suitcases, travel bags, travel pillows, passport holders, travel "
        "adapters, luggage tags."
    ),
    "Jewelry & Accessories": (
        "Rings, necklaces, bracelets, earrings, watches (fashion), handbags, "
        "sunglasses, wallets, keychains."
    ),
    "Office Supplies": (
        "Printer ink, toner, staplers, paper clips, envelopes, filing systems, "
        "whiteboards, dry-erase markers."
    ),
    "Hardware & Tools": (
        "Hammers, screwdrivers, drills, nails, screws, nuts, bolts, paint, "
        "sandpaper, electrical fittings, plumbing supplies."
    ),
    "Services": (
        "Labour charges, delivery fees, installation fees, subscription fees, "
        "repair charges, service charges, consulting."
    ),
    "Other / Uncategorized": (
        "Items that do not clearly fit any other category, or where the item "
        "name is too ambiguous or unclear to categorise reliably."
    ),
}

# ---------------------------------------------------------------------------
# UI configuration
# ---------------------------------------------------------------------------

APP_TITLE: str = "Receipt Intelligence"
APP_SUBTITLE: str = "Turn receipts into structured, categorised data."
APP_ICON: str = "🧾"

# Bulk processing
BULK_MAX_FILES: int = 50
BULK_PROCESSING_DELAY: float = 0.1   # Seconds between receipts to avoid rate-limits

# Debug mode: show OCR text, raw JSON, processing times in expandable panels
DEBUG_MODE: bool = False   # Set True via environment variable RECEIPT_DEBUG=1

def is_debug_mode() -> bool:
    """Check whether debug mode is enabled via env var or config."""
    return os.getenv("RECEIPT_DEBUG", "0").strip() == "1" or DEBUG_MODE
