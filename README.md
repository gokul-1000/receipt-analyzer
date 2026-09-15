# 🧾 Receipt Intelligence

> **AI-powered receipt understanding, item extraction, and product categorisation — built on Streamlit and Groq.**

---

## Problem

Receipts come in dozens of formats: different stores, fonts, layouts, currencies, and quality levels. Manually entering receipt data is tedious and error-prone. Existing solutions either require custom templates per store or blindly trust unreliable OCR output.

## Solution

Receipt Intelligence is an end-to-end AI pipeline that:

1. Accepts any receipt image or PDF
2. Preprocesses and OCRs it robustly
3. Uses a Groq-powered LLM to extract structured data
4. Categorises every item into a predefined taxonomy
5. Validates extracted values arithmetically
6. Presents the results in a polished, interview-ready UI

---

## Architecture

```
Receipt Upload
   ↓ File Validation
   ↓ Image Preprocessing (grayscale, contrast, sharpen, deskew)
   ↓ Tesseract OCR (or PyMuPDF for native-text PDFs)
   ↓ Groq LLM Stage 1: Structured Extraction
   ↓ Deterministic Arithmetic Validation
   ↓ Groq LLM Stage 2: Batch Categorisation
   ↓ Multi-Signal Confidence Scoring
   ↓ Streamlit Dashboard
```

### Project Structure

```
receipt_ai/
├── app.py                     # Streamlit UI entry point
├── config.py                  # All configuration, categories, thresholds
├── requirements.txt
├── packages.txt               # Streamlit Cloud system packages
├── .env.example
│
├── core/
│   ├── pipeline.py            # Pipeline orchestrator
│   ├── models.py              # Pydantic data models
│   ├── categories.py          # Taxonomy helpers + hallucination guard
│   └── validators.py          # Deterministic arithmetic validation
│
├── services/
│   ├── groq_service.py        # Groq API abstraction with retries
│   ├── ocr_service.py         # Tesseract OCR + quality estimation
│   ├── pdf_service.py         # PDF text extraction + page rendering
│   ├── image_service.py       # Image preprocessing pipeline
│   └── categorization_service.py  # Two-stage batch categorisation
│
├── utils/
│   ├── parsing.py             # Robust JSON parsing for LLM output
│   ├── normalization.py       # Price/date/currency normalisation
│   └── confidence.py          # Multi-signal confidence scoring
│
├── prompts/
│   ├── extraction_prompt.py   # Stage 1: Receipt extraction prompt
│   └── categorization_prompt.py   # Stage 2: Categorisation prompt
│
└── tests/
    ├── conftest.py
    ├── test_parser.py
    ├── test_validator.py
    └── test_categories.py
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| UI | Streamlit | Fast prototyping, cloud deployment, native widgets |
| LLM | Groq (openai/gpt-oss-120b) | Fast inference, free tier, excellent JSON output |
| OCR | Tesseract + pytesseract | Best open-source OCR, Streamlit Cloud compatible |
| PDF | PyMuPDF | Zero external deps, fast, handles scanned+native PDFs |
| Image preprocessing | Pillow + NumPy | Grayscale, contrast, deskew, denoise |
| Data models | Pydantic v2 | Strict validation, auto-coercion, clean JSON serialisation |
| Charts | Plotly | Interactive, polished, dark-mode friendly |
| Testing | pytest | Fast, no API key needed (mock-based) |

---

## Why OCR + LLM?

**OCR alone** can't understand receipt structure: it just gives you raw text. It can't tell the difference between a product name, a price, and a subtotal.

**LLM alone** on raw images is slow, expensive, and unreliable for precise numeric extraction.

**OCR + LLM** is the right combination:
- OCR converts the image into clean text (its strength)
- LLM understands the structure and semantics of that text (its strength)
- Result: fast, accurate, explainable extraction at low cost

---

## Why Groq?

- **Speed**: Groq's LPU hardware delivers 10–100× faster inference than typical GPU APIs
- **Free tier**: Perfect for prototyping and interviews
- **Quality**: `openai/gpt-oss-120b` produces excellent structured JSON output
- **Configurable**: Model name stored in `config.py` and overrideable via `GROQ_MODEL` env/secrets — easy to swap

---

## Category System

21 predefined categories stored centrally in `config.py`:

```python
CATEGORY_LIST = [
    "Food & Beverages", "Clothing & Footwear", "Electronics & Accessories",
    "Health & Personal Care", "Beauty & Cosmetics", "Home & Kitchen",
    "Grocery", "Household Supplies", "Furniture", "Sports & Fitness",
    "Books & Stationery", "Toys & Games", "Automotive", "Pet Supplies",
    "Baby Products", "Travel & Luggage", "Jewelry & Accessories",
    "Office Supplies", "Hardware & Tools", "Services", "Other / Uncategorized"
]
```

**Hallucination prevention**: `core/categories.py::sanitize_category()` intercepts every LLM response and replaces any non-canonical category with `"Other / Uncategorized"`. The LLM is physically unable to introduce new categories into the final output.

---

## Validation Strategy

Three levels of deterministic validation (no LLM involved):

1. **Line-item level**: `qty × unit_price ≈ total_price` (±2% tolerance)
2. **Subtotal level**: `Σ(line totals) ≈ stated subtotal` (±5% tolerance)
3. **Total level**: `subtotal + tax − discount ≈ total` (±5% tolerance)

Validation statuses:
- `VALIDATED` — all checks passed
- `PARTIALLY_VALIDATED` — insufficient data for full check, but no errors found
- `WARNING` — minor discrepancies detected
- `FAILED_VALIDATION` — clear arithmetic errors detected

---

## Confidence Scoring

Confidence is **not** a number the LLM makes up. It is computed from four independent signals:

| Signal | Weight | Description |
|---|---|---|
| OCR Quality | 20% | Tesseract per-character confidence mean |
| Structural Completeness | 25% | How many key fields were extracted |
| Arithmetic Validation | 30% | Did the numbers pass validation? |
| Category Confidence | 25% | LLM's self-reported category confidence |

Thresholds (configurable in `config.py`):
- **🟢 High**: ≥ 85%
- **🟡 Medium**: 60–84%
- **🔴 Low**: < 60%

---

## Bulk Processing

Upload up to 50 receipts at once. Each receipt is:
- Processed independently (one failure doesn't stop the batch)
- Assigned a job status (Complete / Error)
- Included in aggregated analytics

Analytics include:
- Total spend and item count
- Spending by category (bar chart)
- Receipts per merchant
- Top purchased items
- Highest-spend items

Export: CSV and JSON download.

---

## Error Handling

Every failure mode returns a friendly user-facing message, never a Python traceback:

| Error | User Message |
|---|---|
| Unsupported file | "Unsupported file type. Please upload JPG, PNG, WEBP, or PDF." |
| Corrupted PDF | "PDF is corrupted or unreadable." |
| Empty/blank image | "No text could be extracted. Try a clearer image." |
| Groq API failure | "AI extraction failed. Check your API key." |
| Rate limit | Automatic retry with exponential back-off + fallback model |
| Malformed LLM JSON | Multi-attempt recovery (fence stripping, brace extraction, fix trailing commas) |
| File too large | "File exceeds maximum size of 20 MB." |

---

## Local Setup

### 1. Install dependencies

```bash
cd receipt_ai
pip install -r requirements.txt
```

### 2. Install Tesseract OCR

**Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki

**macOS**: `brew install tesseract`

**Linux**: `sudo apt-get install tesseract-ocr`

### 3. Configure API key

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Or enter it in the sidebar when the app is running.

### 4. Run the app

```bash
streamlit run app.py
```

---

## Streamlit Cloud Deployment

1. Push this project to a GitHub repository.

2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and connect your repo.

3. Set the main file path to: `receipt_ai/app.py`

4. In the app settings → **Secrets**, add:

```toml
GROQ_API_KEY = "gsk_your_actual_key_here"
```

5. The `packages.txt` file automatically installs Tesseract on the Streamlit Cloud server.

6. Deploy. ✅

---

## Running Tests

```bash
cd receipt_ai
pytest tests/ -v
```

Tests do **not** require a Groq API key. All LLM interactions are tested with mocked data.

---

## Limitations

- **OCR accuracy**: Tesseract struggles with very poor scans, handwritten text, or exotic fonts. For production, consider a vision-capable LLM (e.g., GPT-4o) as a fallback.
- **Multi-language**: Currently optimised for English receipts.
- **Exotic layouts**: Highly unconventional receipt formats may require prompt tuning.
- **Rate limits**: Groq free tier has rate limits; bulk processing includes delays to mitigate this.

---

## Future Improvements

- **Vision fallback**: Use GPT-4o or Gemini Vision when Tesseract confidence is low
- **Merchant-specific parsers**: Custom extraction rules for known store chains
- **Multilingual OCR**: Tesseract supports 100+ languages — expose this in UI
- **Database persistence**: Store and query historical receipt data
- **Human-in-the-loop**: Flag low-confidence items for manual review
- **Model evaluation framework**: A/B test different Groq models on a labelled dataset
- **Receipt deduplication**: Hash-based detection of duplicate uploads
- **Currency conversion**: Unified multi-currency analytics dashboard
