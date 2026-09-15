"""
app.py — Receipt Intelligence Streamlit Application.

This is the entry point.  All UI logic lives here; business logic
is delegated to core/ and services/ modules.

Run locally:
    streamlit run app.py

Deploy on Streamlit Cloud:
    Set GROQ_API_KEY in .streamlit/secrets.toml
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
import traceback
from typing import Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup (needed when running from project root)
# ---------------------------------------------------------------------------
import os
sys.path.insert(0, os.path.dirname(__file__))

# Load .env file for local development (no-op if not present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import (
    APP_ICON,
    APP_SUBTITLE,
    APP_TITLE,
    BULK_MAX_FILES,
    CATEGORY_LIST,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    SUPPORTED_TYPES,
    is_debug_mode,
)
from core.models import (
    BulkReceiptResult,
    ExtractionSource,
    ProcessingStatus,
    Receipt,
    ReceiptItem,
    ValidationStatus,
)
from core.pipeline import PipelineError, process_receipt
from services.groq_service import GroqServiceError, create_groq_service, get_api_key
from utils.confidence import tier_label

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Receipt Intelligence",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
/* Global */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #1a1d27; border-right: 1px solid #2d3149; }

/* Hero */
.hero-title { font-size: 2.8rem; font-weight: 800; color: #ffffff; margin-bottom: 0.2rem; }
.hero-sub { font-size: 1.15rem; color: #8b95b0; margin-bottom: 2rem; }

/* Cards */
.metric-card {
    background: #1e2235;
    border: 1px solid #2d3149;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value { font-size: 1.6rem; font-weight: 700; color: #7c8cf8; }
.metric-label { font-size: 0.8rem; color: #8b95b0; text-transform: uppercase; letter-spacing: 0.05em; }

/* Pipeline steps */
.pipeline-step {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.4rem 0; font-size: 0.95rem; color: #c5cbe0;
}
.step-done { color: #22c55e; }
.step-active { color: #7c8cf8; }

/* Category badge */
.cat-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    background: #2d3149;
    color: #a5b4fc;
    white-space: nowrap;
}

/* Confidence pill */
.conf-high { color: #22c55e; font-weight: 700; }
.conf-medium { color: #facc15; font-weight: 700; }
.conf-low { color: #f87171; font-weight: 700; }

/* Validation status */
.status-validated { color: #22c55e; font-weight: 700; }
.status-partial { color: #facc15; font-weight: 700; }
.status-warning { color: #fb923c; font-weight: 700; }
.status-failed { color: #f87171; font-weight: 700; }

/* Source badge */
.source-ai { color: #7c8cf8; font-size: 0.75rem; }
.source-user { color: #34d399; font-size: 0.75rem; }

/* Section headers */
.section-title {
    font-size: 1.2rem; font-weight: 700; color: #e5e8f0;
    border-bottom: 1px solid #2d3149;
    padding-bottom: 0.4rem; margin-top: 1.5rem; margin-bottom: 1rem;
}

/* Upload area override */
[data-testid="stFileUploader"] {
    border: 2px dashed #2d3149 !important;
    border-radius: 12px !important;
    background: #1a1d27 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _init_session() -> None:
    defaults = {
        "receipt": None,
        "bulk_results": [],
        "edit_mode": False,
        "edited_items": {},
        "upload_mode": "Single Receipt",
        "extraction_mode": "vision",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    """Render sidebar and return the selected upload mode."""
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center; padding: 1rem 0;'>"
            "<span style='font-size:2.5rem'>🧾</span>"
            "<h2 style='color:#ffffff; margin:0; font-size:1.4rem;'>Receipt Intelligence</h2>"
            "<p style='color:#8b95b0; font-size:0.8rem; margin:0;'>AI-Powered Receipt Analysis</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        mode = st.radio(
            "**Upload Mode**",
            ["Single Receipt", "Bulk Analysis"],
            index=0 if st.session_state.upload_mode == "Single Receipt" else 1,
        )
        st.session_state.upload_mode = mode

        st.divider()
        st.markdown("**⚡ Extraction Mode**")
        ext_choice = st.radio(
            "Extraction Mode",
            ["Vision AI (Recommended)", "OCR + LLM"],
            index=0 if st.session_state.get("extraction_mode", "vision") == "vision" else 1,
            help="Vision AI uses Qwen 3.6 27B Vision directly on Groq. OCR + LLM uses Tesseract OCR followed by LLM extraction.",
        )
        st.session_state.extraction_mode = "vision" if "Vision AI" in ext_choice else "ocr_llm"

        st.divider()
        st.markdown("**⚙️ Settings**")

        # API key input (local dev helper)
        api_key = get_api_key()
        if not api_key:
            st.warning("⚠️ GROQ_API_KEY not found.")
            entered_key = st.text_input(
                "Enter Groq API Key",
                type="password",
                placeholder="gsk_...",
                help="Add to .env or Streamlit secrets for production.",
            )
            if entered_key:
                os.environ["GROQ_API_KEY"] = entered_key
                st.success("Key saved for this session.")

        debug = st.checkbox("🔬 Developer Mode", value=is_debug_mode())
        if debug:
            os.environ["RECEIPT_DEBUG"] = "1"

        st.divider()
        st.markdown("**📦 Category System**")
        with st.expander("View all categories", expanded=False):
            for cat in CATEGORY_LIST:
                st.markdown(f"• {cat}")

        st.divider()
        _render_how_it_works()

        return mode


def _render_how_it_works() -> None:
    with st.expander("❓ How It Works", expanded=False):
        st.markdown(
            """
1. **Upload** — JPG, PNG, WEBP, or PDF  
2. **OCR** — Tesseract extracts text with image preprocessing  
3. **AI Extraction** — Groq LLM parses receipt structure  
4. **Categorisation** — Batch category assignment  
5. **Validation** — Arithmetic cross-checks  
6. **Results** — Structured data with confidence scores
"""
        )
    with st.expander("🏗️ Architecture", expanded=False):
        st.code(
            """Receipt Upload
   ↓ File Validation
   ↓ Image Preprocessing
   ↓ Tesseract OCR
   ↓ Groq LLM (Stage 1: Extract)
   ↓ Deterministic Validation
   ↓ Groq LLM (Stage 2: Categorise)
   ↓ Confidence Scoring
   ↓ Streamlit Dashboard""",
            language="text",
        )


# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------


def render_hero() -> None:
    st.markdown(
        f"<h1 class='hero-title'>{APP_ICON} {APP_TITLE}</h1>"
        f"<p class='hero-sub'>{APP_SUBTITLE}</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pipeline progress indicator
# ---------------------------------------------------------------------------


def render_pipeline_progress(step: int, mode: str = "vision") -> None:
    """Display a visual pipeline with completed / active steps."""
    if mode == "vision":
        steps = [
            "Document received",
            "Image prepared",
            "Vision AI analyzed",
            "Items extracted",
            "Categories assigned",
            "Validation completed",
        ]
    else:
        steps = [
            "Document received",
            "OCR completed",
            "Receipt parsed",
            "Items extracted",
            "Categories assigned",
            "Validation completed",
        ]
    cols = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps)):
        with col:
            if i < step:
                st.markdown(f"✅ **{label}**")
            elif i == step:
                st.markdown(f"⏳ *{label}...*")
            else:
                st.markdown(f"⬜ {label}")


# ---------------------------------------------------------------------------
# Validation badge
# ---------------------------------------------------------------------------


def _validation_badge(status: ValidationStatus) -> str:
    mapping = {
        ValidationStatus.VALIDATED: ("✅ VALIDATED", "status-validated"),
        ValidationStatus.PARTIALLY_VALIDATED: ("🟡 PARTIAL", "status-partial"),
        ValidationStatus.WARNING: ("⚠️ WARNING", "status-warning"),
        ValidationStatus.FAILED_VALIDATION: ("❌ FAILED", "status-failed"),
        ValidationStatus.NOT_CHECKED: ("— N/A", ""),
    }
    text, css = mapping.get(status, ("—", ""))
    return f"<span class='{css}'>{text}</span>"


def _confidence_badge(score: float) -> str:
    label = tier_label(score)
    pct = f"{score:.0%}"
    if score >= CONFIDENCE_HIGH:
        return f"<span class='conf-high'>{label} ({pct})</span>"
    if score >= CONFIDENCE_MEDIUM:
        return f"<span class='conf-medium'>{label} ({pct})</span>"
    return f"<span class='conf-low'>{label} ({pct})</span>"


# ---------------------------------------------------------------------------
# Single receipt results
# ---------------------------------------------------------------------------


def render_single_receipt(receipt: Receipt, file_bytes: bytes, filename: str) -> None:
    """Render the full single-receipt results view."""
    # Layout: left = preview, right = data
    col_preview, col_data = st.columns([1, 1.6], gap="large")

    with col_preview:
        _render_preview(file_bytes, filename, receipt)

    with col_data:
        _render_receipt_summary(receipt)
        _render_items_table(receipt)
        _render_validation_panel(receipt)

    # Debug panel
    if is_debug_mode():
        _render_debug_panel(receipt)


def _render_preview(file_bytes: bytes, filename: str, receipt: Receipt) -> None:
    st.markdown("<div class='section-title'>📄 Receipt Preview</div>", unsafe_allow_html=True)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        from services.pdf_service import render_pdf_page_preview
        preview = render_pdf_page_preview(file_bytes, page_idx=0)
        if preview:
            st.image(preview, caption=f"Page 1 of {receipt.page_count}", use_container_width=True)
        else:
            st.info("PDF preview not available.")
    else:
        st.image(file_bytes, caption=filename, use_container_width=True)

    with st.expander("🔤 View Extracted Text / Log"):
        st.text_area("Extraction Log / OCR Output", value=receipt.ocr_text, height=300, disabled=True)


def _render_receipt_summary(receipt: Receipt) -> None:
    st.markdown("<div class='section-title'>🏪 Receipt Summary</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("Merchant", receipt.merchant_name or "Unknown")
    c2.metric("Date", receipt.receipt_date or "—")

    c3, c4, c5 = st.columns(3)
    currency = receipt.currency or ""
    c3.metric("Subtotal", f"{currency} {receipt.subtotal:.2f}" if receipt.subtotal else "—")
    c4.metric("Tax", f"{currency} {receipt.tax:.2f}" if receipt.tax else "—")
    c5.metric("Total", f"{currency} {receipt.total:.2f}" if receipt.total else "—")

    c6, c7, c8 = st.columns(3)
    c6.metric("Items", receipt.item_count)
    with c7:
        st.markdown("**Validation**")
        st.markdown(
            _validation_badge(receipt.validation.status),
            unsafe_allow_html=True,
        )
    with c8:
        st.markdown("**Extraction Mode**")
        is_vision = "[Direct Vision AI" in (receipt.ocr_text or "")
        mode_label = "👁️ Vision AI" if is_vision else "📄 OCR + LLM"
        st.markdown(f"<span class='cat-badge'>{mode_label}</span>", unsafe_allow_html=True)

    # Overall confidence
    st.markdown(
        f"**Overall Confidence:** {_confidence_badge(receipt.overall_confidence)}",
        unsafe_allow_html=True,
    )
    st.progress(receipt.overall_confidence)


def _render_items_table(receipt: Receipt) -> None:
    st.markdown("<div class='section-title'>🛒 Purchased Items</div>", unsafe_allow_html=True)

    if not receipt.items:
        st.warning("No items were extracted from this receipt.")
        return

    # Edit mode toggle
    edit_col, _ = st.columns([1, 3])
    with edit_col:
        edit_mode = st.toggle("✏️ Edit Results", value=st.session_state.edit_mode)
        st.session_state.edit_mode = edit_mode

    if edit_mode:
        _render_editable_items(receipt)
    else:
        _render_read_only_items(receipt)


def _render_read_only_items(receipt: Receipt) -> None:
    currency = receipt.currency or ""
    rows = []
    for item in receipt.items:
        edit_info = st.session_state.edited_items.get(item.name, {})
        effective_item = _apply_edits(item, edit_info)

        source_html = (
            "<span class='source-user'>✏ User</span>"
            if edit_info
            else "<span class='source-ai'>🤖 AI</span>"
        )
        rows.append(
            {
                "Item": effective_item.name,
                "Qty": effective_item.quantity or "—",
                "Unit Price": (
                    f"{currency}{effective_item.unit_price:.2f}"
                    if effective_item.unit_price is not None
                    else "—"
                ),
                "Total": (
                    f"{currency}{effective_item.total_price:.2f}"
                    if effective_item.total_price is not None
                    else "—"
                ),
                "Category": effective_item.category,
                "Confidence": f"{effective_item.extraction_confidence:.0%}",
                "Source": "User" if edit_info else "AI",
            }
        )

    df = pd.DataFrame(rows)

    def colour_confidence(val: str) -> str:
        try:
            pct = float(val.strip("%")) / 100
        except (ValueError, AttributeError):
            return ""
        if pct >= CONFIDENCE_HIGH:
            return "color: #22c55e"
        if pct >= CONFIDENCE_MEDIUM:
            return "color: #facc15"
        return "color: #f87171"

    styled = df.style.map(colour_confidence, subset=["Confidence"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Category reasoning (expandable)
    with st.expander("💬 Category Reasoning"):
        for item in receipt.items:
            st.markdown(
                f"**{item.name}** → `{item.category}` "
                f"({item.category_confidence:.0%}): *{item.category_reasoning or 'N/A'}*"
            )

    # Download section
    _render_download_buttons(receipt)


def _render_editable_items(receipt: Receipt) -> None:
    st.info("Edit the fields below, then click **Save** to re-run validation.")
    currency = receipt.currency or ""

    for idx, item in enumerate(receipt.items):
        edit_info = st.session_state.edited_items.get(item.name, {})
        with st.expander(
            f"**{edit_info.get('name', item.name)}** — {item.category}", expanded=False
        ):
            e1, e2, e3, e4, e5 = st.columns(5)
            with e1:
                new_name = st.text_input("Name", value=edit_info.get("name", item.name), key=f"name_{idx}")
            with e2:
                qty_val = edit_info.get("quantity", item.quantity)
                new_qty = st.number_input("Qty", value=float(qty_val) if qty_val else 0.0, min_value=0.0, key=f"qty_{idx}")
            with e3:
                up_val = edit_info.get("unit_price", item.unit_price)
                new_up = st.number_input("Unit Price", value=float(up_val) if up_val else 0.0, min_value=0.0, key=f"up_{idx}")
            with e4:
                tp_val = edit_info.get("total_price", item.total_price)
                new_tp = st.number_input("Total", value=float(tp_val) if tp_val else 0.0, min_value=0.0, key=f"tp_{idx}")
            with e5:
                current_cat = edit_info.get("category", item.category)
                cat_idx = CATEGORY_LIST.index(current_cat) if current_cat in CATEGORY_LIST else 0
                new_cat = st.selectbox("Category", CATEGORY_LIST, index=cat_idx, key=f"cat_{idx}")

            if st.button("💾 Save", key=f"save_{idx}"):
                st.session_state.edited_items[item.name] = {
                    "name": new_name,
                    "quantity": new_qty or None,
                    "unit_price": new_up or None,
                    "total_price": new_tp or None,
                    "category": new_cat,
                    "source": ExtractionSource.USER,
                }
                # Mark item as user-corrected
                item.source = ExtractionSource.USER
                st.success(f"✅ Saved changes for '{new_name}'.")
                st.rerun()


def _apply_edits(item: ReceiptItem, edits: dict) -> ReceiptItem:
    """Return a copy of *item* with user edits applied."""
    if not edits:
        return item
    import copy
    new_item = copy.deepcopy(item)
    for field in ("name", "quantity", "unit_price", "total_price", "category"):
        if field in edits and edits[field] is not None:
            setattr(new_item, field, edits[field])
    new_item.source = ExtractionSource.USER
    return new_item


def _render_validation_panel(receipt: Receipt) -> None:
    st.markdown("<div class='section-title'>✅ Validation Results</div>", unsafe_allow_html=True)
    val = receipt.validation

    cols = st.columns(4)
    cols[0].markdown(
        f"**Line Items**<br>{'✅ OK' if val.line_items_ok else '❌ Issues'}", unsafe_allow_html=True
    )
    cols[1].markdown(
        f"**Subtotal Match**<br>{'✅ Yes' if val.subtotal_matches else '⚠️ No'}", unsafe_allow_html=True
    )
    cols[2].markdown(
        f"**Tax Detected**<br>{'✅ Yes' if val.tax_detected else '— No'}", unsafe_allow_html=True
    )
    cols[3].markdown(
        f"**Total Consistent**<br>{'✅ Yes' if val.total_consistent else '⚠️ No'}", unsafe_allow_html=True
    )

    for note in val.notes:
        if note.startswith("✓"):
            st.success(note)
        elif note.startswith("⚠") or note.startswith("ℹ"):
            st.warning(note)
        else:
            st.info(note)


def _render_debug_panel(receipt: Receipt) -> None:
    st.divider()
    st.markdown("### 🔬 Developer Debug Panel")
    c1, c2 = st.columns(2)
    with c1:
        with st.expander("📋 Raw OCR Text"):
            st.text(receipt.ocr_text)
        with st.expander("📊 Validation Detail"):
            st.json(receipt.validation.model_dump())
    with c2:
        with st.expander("🗂 Structured JSON"):
            st.json(receipt.model_dump())
        with st.expander("⏱ Processing Metadata"):
            st.json(
                {
                    "filename": receipt.filename,
                    "processing_time_s": receipt.processing_time_seconds,
                    "ocr_quality": receipt.ocr_quality_score,
                    "overall_confidence": receipt.overall_confidence,
                    "page_count": receipt.page_count,
                }
            )


# ---------------------------------------------------------------------------
# Download buttons
# ---------------------------------------------------------------------------


def _render_download_buttons(receipt: Receipt) -> None:
    st.markdown("**📥 Download Results**")
    c1, c2 = st.columns(2)

    with c1:
        json_bytes = receipt.model_dump_json(indent=2).encode()
        st.download_button(
            "⬇ Download JSON",
            data=json_bytes,
            file_name=f"{receipt.filename or 'receipt'}_results.json",
            mime="application/json",
        )

    with c2:
        csv_bytes = _receipt_to_csv(receipt).encode()
        st.download_button(
            "⬇ Download CSV",
            data=csv_bytes,
            file_name=f"{receipt.filename or 'receipt'}_results.csv",
            mime="text/csv",
        )


def _receipt_to_csv(receipt: Receipt) -> str:
    rows = []
    for item in receipt.items:
        rows.append(
            {
                "receipt_filename": receipt.filename,
                "merchant": receipt.merchant_name or "",
                "date": receipt.receipt_date or "",
                "currency": receipt.currency or "",
                "item_name": item.name,
                "quantity": item.quantity or "",
                "unit_price": item.unit_price or "",
                "total_price": item.total_price or "",
                "category": item.category,
                "category_confidence": f"{item.category_confidence:.2f}",
                "extraction_confidence": f"{item.extraction_confidence:.2f}",
                "validation_status": receipt.validation.status.value,
                "source": item.source.value,
            }
        )
    return pd.DataFrame(rows).to_csv(index=False)


# ---------------------------------------------------------------------------
# Bulk analysis
# ---------------------------------------------------------------------------


def render_bulk_upload() -> None:
    st.markdown("<div class='section-title'>📦 Bulk Receipt Analysis</div>", unsafe_allow_html=True)
    st.info(
        f"Upload up to {BULK_MAX_FILES} receipts at once. "
        "Each receipt is processed independently — one failure won't stop the batch."
    )

    uploaded_files = st.file_uploader(
        "Upload receipts",
        type=list(SUPPORTED_TYPES),
        accept_multiple_files=True,
        help=f"Supported: {', '.join(SUPPORTED_TYPES).upper()}",
    )

    if not uploaded_files:
        return

    if len(uploaded_files) > BULK_MAX_FILES:
        st.error(f"Too many files. Maximum is {BULK_MAX_FILES}.")
        return

    if st.button("🚀 Run Bulk Analysis", type="primary"):
        _run_bulk_analysis(uploaded_files)


def _run_bulk_analysis(uploaded_files: list) -> None:
    """Process multiple receipts sequentially with a live progress UI."""
    api_key = get_api_key()
    if not api_key:
        st.error("⚠️ Groq API key not found. Please add it in the sidebar or secrets.")
        return

    try:
        groq = create_groq_service()
    except GroqServiceError as exc:
        st.error(f"Failed to initialise AI service: {exc}")
        return

    results: list[BulkReceiptResult] = []
    progress_bar = st.progress(0)
    status_container = st.container()

    for i, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name
        result = BulkReceiptResult(filename=filename, status=ProcessingStatus.PROCESSING)

        with status_container:
            st.markdown(f"⏳ Processing **{filename}**...")

        start = time.time()
        try:
            file_bytes = uploaded_file.read()
            mode = st.session_state.get("extraction_mode", "vision")
            receipt = process_receipt(file_bytes, filename, groq, extraction_mode=mode)
            result.receipt = receipt
            result.status = ProcessingStatus.COMPLETE
            result.processing_time_seconds = receipt.processing_time_seconds
        except PipelineError as exc:
            result.status = ProcessingStatus.ERROR
            result.error_message = str(exc)
            result.processing_time_seconds = round(time.time() - start, 2)
            logger.warning("Bulk item %s failed: %s", filename, exc)
        except Exception as exc:
            result.status = ProcessingStatus.ERROR
            result.error_message = "Unexpected error during processing."
            result.processing_time_seconds = round(time.time() - start, 2)
            logger.exception("Unexpected bulk error for %s: %s", filename, exc)

        results.append(result)
        progress_bar.progress((i + 1) / len(uploaded_files))
        time.sleep(0.1)  # Small delay to respect rate limits

    st.session_state.bulk_results = results
    status_container.empty()
    progress_bar.empty()
    st.success(f"✅ Bulk analysis complete: {len(results)} receipts processed.")


def render_bulk_results(results: list[BulkReceiptResult]) -> None:
    if not results:
        return

    st.markdown("<div class='section-title'>📊 Bulk Analysis Results</div>", unsafe_allow_html=True)

    # --- Job status table ---
    _render_bulk_status_table(results)

    # --- Aggregate metrics ---
    successful = [r for r in results if r.status == ProcessingStatus.COMPLETE and r.receipt]
    if not successful:
        st.warning("No receipts were successfully processed.")
        return

    _render_aggregate_metrics(successful)
    _render_category_breakdown(successful)
    _render_merchant_breakdown(successful)
    _render_top_items(successful)
    _render_bulk_export(results)


def _render_bulk_status_table(results: list[BulkReceiptResult]) -> None:
    rows = []
    for r in results:
        status_icon = {
            ProcessingStatus.COMPLETE: "✅",
            ProcessingStatus.ERROR: "❌",
            ProcessingStatus.PROCESSING: "⏳",
            ProcessingStatus.PENDING: "⬜",
        }.get(r.status, "—")

        rows.append(
            {
                "Status": f"{status_icon} {r.status.value.capitalize()}",
                "File": r.filename,
                "Merchant": r.merchant,
                "Date": r.date_str,
                "Items": r.item_count,
                "Total": f"{r.total:.2f}" if r.total else "—",
                "Validation": r.validation_status.value if r.validation_status else "—",
                "Time (s)": f"{r.processing_time_seconds:.1f}",
                "Error": r.error_message or "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_aggregate_metrics(results: list[BulkReceiptResult]) -> None:
    st.markdown("**📈 Aggregate Analytics**")

    total_spend = sum(r.total or 0 for r in results)
    total_items = sum(r.item_count for r in results)
    avg_spend = total_spend / len(results) if results else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Receipts", len(results))
    c2.metric("Total Items", total_items)
    c3.metric("Total Spend", f"{total_spend:,.2f}")
    c4.metric("Avg Receipt Value", f"{avg_spend:,.2f}")


def _render_category_breakdown(results: list[BulkReceiptResult]) -> None:
    st.markdown("**🏷️ Spending by Category**")

    cat_spend: dict[str, float] = {}
    for r in results:
        if r.receipt:
            for item in r.receipt.items:
                if item.total_price:
                    cat_spend[item.category] = cat_spend.get(item.category, 0) + item.total_price

    if not cat_spend:
        return

    try:
        import plotly.express as px
        df = pd.DataFrame(
            {"Category": list(cat_spend.keys()), "Spend": list(cat_spend.values())}
        ).sort_values("Spend", ascending=False)

        fig = px.bar(
            df,
            x="Spend",
            y="Category",
            orientation="h",
            color="Spend",
            color_continuous_scale="Blues",
            title="Spending by Category",
        )
        fig.update_layout(
            paper_bgcolor="#0f1117",
            plot_bgcolor="#1a1d27",
            font_color="#c5cbe0",
            title_font_size=14,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        # Fallback to st.bar_chart
        df = pd.DataFrame(cat_spend.items(), columns=["Category", "Spend"])
        st.bar_chart(df.set_index("Category"))


def _render_merchant_breakdown(results: list[BulkReceiptResult]) -> None:
    st.markdown("**🏪 Receipts per Merchant**")
    merchant_count: dict[str, int] = {}
    merchant_spend: dict[str, float] = {}
    for r in results:
        m = r.merchant
        merchant_count[m] = merchant_count.get(m, 0) + 1
        merchant_spend[m] = merchant_spend.get(m, 0) + (r.total or 0)

    df = pd.DataFrame(
        {
            "Merchant": list(merchant_count.keys()),
            "Receipts": list(merchant_count.values()),
            "Total Spend": [merchant_spend.get(m, 0) for m in merchant_count],
        }
    ).sort_values("Receipts", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_top_items(results: list[BulkReceiptResult]) -> None:
    c1, c2 = st.columns(2)

    # Top items by frequency
    item_freq: dict[str, int] = {}
    item_spend: dict[str, float] = {}
    for r in results:
        if r.receipt:
            for item in r.receipt.items:
                name = item.name.lower()
                item_freq[name] = item_freq.get(name, 0) + 1
                item_spend[name] = item_spend.get(name, 0) + (item.total_price or 0)

    with c1:
        st.markdown("**🔄 Most Purchased Items (by frequency)**")
        top_freq = sorted(item_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_freq:
            df = pd.DataFrame(top_freq, columns=["Item", "Count"])
            st.dataframe(df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**💰 Highest Spend Items**")
        top_spend = sorted(item_spend.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_spend:
            df = pd.DataFrame(top_spend, columns=["Item", "Total Spend"])
            df["Total Spend"] = df["Total Spend"].map("{:.2f}".format)
            st.dataframe(df, use_container_width=True, hide_index=True)


def _render_bulk_export(results: list[BulkReceiptResult]) -> None:
    st.markdown("<div class='section-title'>📥 Export Bulk Results</div>", unsafe_allow_html=True)

    # Build flat CSV
    rows = []
    for r in results:
        if r.receipt:
            for item in r.receipt.items:
                rows.append(
                    {
                        "receipt_filename": r.filename,
                        "merchant": r.receipt.merchant_name or "",
                        "date": r.receipt.receipt_date or "",
                        "currency": r.receipt.currency or "",
                        "item_name": item.name,
                        "quantity": item.quantity or "",
                        "unit_price": item.unit_price or "",
                        "total_price": item.total_price or "",
                        "category": item.category,
                        "category_confidence": f"{item.category_confidence:.2f}",
                        "extraction_confidence": f"{item.extraction_confidence:.2f}",
                        "validation_status": r.receipt.validation.status.value,
                        "processing_status": r.status.value,
                        "error_message": r.error_message or "",
                    }
                )
        else:
            rows.append(
                {
                    "receipt_filename": r.filename,
                    "merchant": "",
                    "date": "",
                    "currency": "",
                    "item_name": "",
                    "quantity": "",
                    "unit_price": "",
                    "total_price": "",
                    "category": "",
                    "category_confidence": "",
                    "extraction_confidence": "",
                    "validation_status": "",
                    "processing_status": r.status.value,
                    "error_message": r.error_message or "",
                }
            )

    c1, c2 = st.columns(2)
    if rows:
        csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode()
        with c1:
            st.download_button(
                "⬇ Download CSV",
                data=csv_bytes,
                file_name="bulk_receipt_analysis.csv",
                mime="text/csv",
            )

        json_results = [r.model_dump() for r in results]
        json_bytes = json.dumps(json_results, indent=2, default=str).encode()
        with c2:
            st.download_button(
                "⬇ Download JSON",
                data=json_bytes,
                file_name="bulk_receipt_analysis.json",
                mime="application/json",
            )


# ---------------------------------------------------------------------------
# Single receipt upload and processing
# ---------------------------------------------------------------------------


def render_single_upload() -> None:
    st.markdown("<div class='section-title'>📤 Upload Receipt</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag & drop a receipt here",
        type=list(SUPPORTED_TYPES),
        help=f"Supported: {', '.join(SUPPORTED_TYPES).upper()}. Max {20}MB.",
        label_visibility="collapsed",
    )

    st.caption(f"Supported: {' • '.join(t.upper() for t in SUPPORTED_TYPES)}")

    if not uploaded:
        _render_welcome_placeholder()
        return

    file_bytes = uploaded.read()
    filename = uploaded.name

    # Check API key
    api_key = get_api_key()
    if not api_key:
        st.error(
            "⚠️ Groq API key not found. "
            "Add it in the sidebar or set GROQ_API_KEY in your .env file."
        )
        return

    # Process only if file changed or extraction mode changed
    ext_mode = st.session_state.get("extraction_mode", "vision")
    cache_key = f"receipt_{hash(file_bytes)}_{ext_mode}"
    if st.session_state.get("_last_cache_key") != cache_key:
        st.session_state._last_cache_key = cache_key
        st.session_state.receipt = None
        st.session_state.edited_items = {}
        st.session_state.edit_mode = False

        mode_desc = "Vision AI" if ext_mode == "vision" else "OCR + LLM"
        with st.spinner(f"🔄 Analysing receipt with {mode_desc}..."):
            _run_single_pipeline(file_bytes, filename, ext_mode)

    receipt: Optional[Receipt] = st.session_state.receipt
    if receipt:
        render_single_receipt(receipt, file_bytes, filename)


def _run_single_pipeline(file_bytes: bytes, filename: str, extraction_mode: str = "vision") -> None:
    """Run the processing pipeline and store result in session state."""
    # Pipeline progress display
    progress_placeholder = st.empty()

    def update_progress(step: int):
        with progress_placeholder.container():
            render_pipeline_progress(step, mode=extraction_mode)

    update_progress(0)

    try:
        groq = create_groq_service()
    except GroqServiceError as exc:
        st.error(f"⚠️ Failed to initialise AI service: {exc}")
        progress_placeholder.empty()
        return

    update_progress(1)

    try:
        receipt = process_receipt(file_bytes, filename, groq, extraction_mode=extraction_mode)
        update_progress(5)
        time.sleep(0.5)
        progress_placeholder.empty()
        st.session_state.receipt = receipt
    except PipelineError as exc:
        progress_placeholder.empty()
        st.error(f"⚠️ {exc}")
        logger.warning("Pipeline error for %s: %s", filename, exc)
    except Exception as exc:
        progress_placeholder.empty()
        st.error(
            "⚠️ An unexpected error occurred. "
            "Please try a different file or check your API key."
        )
        logger.exception("Unexpected error for %s: %s", filename, exc)


def _render_welcome_placeholder() -> None:
    st.markdown(
        """
<div style='text-align:center; padding: 4rem 2rem; color: #8b95b0;'>
    <div style='font-size: 3rem; margin-bottom: 1rem;'>🧾</div>
    <h3 style='color: #c5cbe0;'>Upload a receipt to get started</h3>
    <p>Supports JPG, PNG, WEBP, and PDF formats.<br>
    Works with different stores, layouts, and currencies.</p>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _init_session()

    mode = render_sidebar()
    render_hero()

    if mode == "Single Receipt":
        render_single_upload()
    else:
        render_bulk_upload()
        if st.session_state.bulk_results:
            render_bulk_results(st.session_state.bulk_results)


if __name__ == "__main__":
    main()
