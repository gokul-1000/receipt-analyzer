"""
core/models.py — Pydantic data models for the Receipt Intelligence system.

All internal data structures are defined here.  Using Pydantic v2 ensures:
- strict type coercion and validation
- easy JSON serialisation / deserialisation
- clear data contracts between pipeline stages
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ValidationStatus(str, Enum):
    VALIDATED = "VALIDATED"
    PARTIALLY_VALIDATED = "PARTIALLY_VALIDATED"
    WARNING = "WARNING"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    NOT_CHECKED = "NOT_CHECKED"


class ConfidenceTier(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


class ExtractionSource(str, Enum):
    AI = "AI"
    USER = "User"


# ---------------------------------------------------------------------------
# Line-item validation detail
# ---------------------------------------------------------------------------


class LineItemValidation(BaseModel):
    """Arithmetic check result for a single receipt line."""

    is_valid: bool = True
    expected_total: Optional[float] = None
    actual_total: Optional[float] = None
    discrepancy: Optional[float] = None
    note: str = ""


# ---------------------------------------------------------------------------
# Receipt item
# ---------------------------------------------------------------------------


class ReceiptItem(BaseModel):
    """One purchased item extracted from a receipt."""

    name: str = Field(..., description="Product / item name as it appears on the receipt")
    quantity: Optional[float] = Field(None, description="Quantity purchased; null if not stated")
    unit_price: Optional[float] = Field(None, description="Price per unit; null if not stated")
    total_price: Optional[float] = Field(None, description="Total price for this line; null if unreadable")
    currency: Optional[str] = Field(None, description="Currency code or symbol, e.g. USD, ₹, £")

    # Category
    category: str = Field("Other / Uncategorized", description="Predefined category")
    category_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Categorisation confidence 0-1")
    category_reasoning: str = Field("", description="Brief explanation for category assignment")

    # Extraction quality
    extraction_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Overall extraction confidence 0-1")
    is_uncertain: bool = Field(False, description="True when OCR / extraction quality is poor")
    original_text: str = Field("", description="Raw OCR text that produced this item")

    # Validation
    line_validation: LineItemValidation = Field(default_factory=LineItemValidation)

    # Edit tracking
    source: ExtractionSource = Field(ExtractionSource.AI, description="AI or User-corrected")

    @field_validator("quantity", "unit_price", "total_price", mode="before")
    @classmethod
    def coerce_numeric(cls, v: Any) -> Optional[float]:
        """Accept numeric strings; return None for empty/null/zero-division."""
        if v is None or v == "" or v == "null":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @field_validator("category_confidence", "extraction_confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, f))

    @property
    def confidence_tier(self) -> ConfidenceTier:
        from config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM  # avoid circular at module level
        score = self.extraction_confidence
        if score >= CONFIDENCE_HIGH:
            return ConfidenceTier.HIGH
        if score >= CONFIDENCE_MEDIUM:
            return ConfidenceTier.MEDIUM
        return ConfidenceTier.LOW


# ---------------------------------------------------------------------------
# Receipt-level validation result
# ---------------------------------------------------------------------------


class ReceiptValidation(BaseModel):
    """Aggregated validation result for the full receipt."""

    status: ValidationStatus = ValidationStatus.NOT_CHECKED
    line_items_ok: bool = False
    subtotal_matches: bool = False
    tax_detected: bool = False
    total_consistent: bool = False
    computed_subtotal: Optional[float] = None
    receipt_subtotal: Optional[float] = None
    receipt_total: Optional[float] = None
    discrepancy: Optional[float] = None
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Full receipt model
# ---------------------------------------------------------------------------


class Receipt(BaseModel):
    """Complete parsed and validated receipt."""

    # Metadata
    merchant_name: Optional[str] = None
    receipt_date: Optional[str] = None          # ISO string or original string
    currency: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    discount: Optional[float] = None
    total: Optional[float] = None
    payment_method: Optional[str] = None

    # Items
    items: list[ReceiptItem] = Field(default_factory=list)

    # Validation
    validation: ReceiptValidation = Field(default_factory=ReceiptValidation)

    # Processing metadata
    ocr_text: str = ""
    ocr_quality_score: float = 0.5   # 0-1 estimate of OCR readability
    overall_confidence: float = 0.5
    processing_time_seconds: float = 0.0
    filename: str = ""
    page_count: int = 1

    @field_validator("subtotal", "tax", "discount", "total", mode="before")
    @classmethod
    def coerce_numeric(cls, v: Any) -> Optional[float]:
        if v is None or v == "" or v == "null":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @property
    def item_count(self) -> int:
        return len(self.items)

    @property
    def computed_total(self) -> Optional[float]:
        totals = [i.total_price for i in self.items if i.total_price is not None]
        return round(sum(totals), 4) if totals else None


# ---------------------------------------------------------------------------
# Bulk processing result for a single file
# ---------------------------------------------------------------------------


class BulkReceiptResult(BaseModel):
    """Processing outcome for one receipt in a bulk batch."""

    filename: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    receipt: Optional[Receipt] = None
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0

    @property
    def merchant(self) -> str:
        if self.receipt and self.receipt.merchant_name:
            return self.receipt.merchant_name
        return "Unknown"

    @property
    def date_str(self) -> str:
        if self.receipt and self.receipt.receipt_date:
            return str(self.receipt.receipt_date)
        return "—"

    @property
    def total(self) -> Optional[float]:
        return self.receipt.total if self.receipt else None

    @property
    def item_count(self) -> int:
        return self.receipt.item_count if self.receipt else 0

    @property
    def validation_status(self) -> ValidationStatus:
        if self.receipt:
            return self.receipt.validation.status
        return ValidationStatus.NOT_CHECKED


# ---------------------------------------------------------------------------
# OCR result
# ---------------------------------------------------------------------------


class OCRResult(BaseModel):
    """Output of the OCR stage."""

    text: str = ""
    quality_score: float = 0.5   # 0-1 heuristic estimate
    page_count: int = 1
    is_selectable_text: bool = False   # True when PDF had embedded text
    method_used: str = "tesseract"
    warnings: list[str] = Field(default_factory=list)
