"""
summariser.py — Generate retrieval-optimised summaries for tables and images.

IMPORTANT: Text chunks are NOT summarised — they are embedded directly.
Only tables and images go through Gemini summarisation before embedding,
because raw table HTML and raw base64 bytes embed poorly as vectors.

Uses the new google-genai SDK (replaces deprecated google-generativeai).
"""

import base64
from typing import List

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not set.\n"
        "Add  GEMINI_API_KEY=your_key  to your .env file.\n"
        "Get a key at: https://aistudio.google.com/apikey"
    )

# New SDK — single client object, no global configure() call needed
_client = genai.Client(api_key=GEMINI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_TABLE_PROMPT = (
    "You are an assistant tasked with summarizing tables for retrieval. "
    "These summaries will be embedded and used to retrieve the raw table elements. "
    "Give a concise summary of the table that is well optimized for retrieval.\n\n"
    "Table:\n{table}"
)

_IMAGE_PROMPT = (
    "You are an assistant tasked with summarizing images for retrieval. "
    "These summaries will be embedded and used to retrieve the raw image. "
    "Give a concise summary of the image that is well optimized for retrieval."
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _call(contents) -> str:
    """Call Gemini and return the text response, or empty string on failure."""
    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
        )
        return response.text.strip()
    except Exception as exc:
        print(f"  [summariser] Gemini error: {exc}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def summarise_tables(tables: List[str]) -> List[str]:
    """
    One retrieval-optimised summary per table.
    Raw table strings are stored separately and returned at retrieval time.
    """
    summaries = []
    for i, table in enumerate(tables):
        print(f"  Summarising table {i + 1}/{len(tables)}…")
        summary = _call(_TABLE_PROMPT.format(table=table))
        summaries.append(summary or table[:500])   # fallback: first 300 chars
    return summaries


def summarise_images(b64_images: List[str]) -> List[str]:
    """
    One retrieval-optimised text summary per image.
    Gemini is natively multimodal — image passed as inline bytes.
    Raw base64 strings stored separately and returned at retrieval time.
    """
    summaries = []
    for i, b64 in enumerate(b64_images):
        print(f"  Summarising image {i + 1}/{len(b64_images)}…")
        image_part = types.Part.from_bytes(
            data=base64.b64decode(b64),
            mime_type="image/jpeg",
        )
        summary = _call([_IMAGE_PROMPT, image_part])
        summaries.append(summary or f"Image {i + 1}")
    return summaries