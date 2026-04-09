"""
chat_engine.py — Build a multimodal prompt from retrieved raw elements
                 and call Gemini via the new google-genai SDK.

Retrieved elements can be any mix of:
  • text chunks   → added as plain strings
  • table strings → added as plain strings
  • base64 images → resized (1300×600) then added as inline image parts
.
"""

import base64
import io
from typing import Dict, List

from google import genai
from google.genai import types
from PIL import Image as PILImage

from config import GEMINI_API_KEY, GEMINI_MODEL, TOP_K
from store import MultiVectorStore, RawElement


# ─────────────────────────────────────────────────────────────────────────────
# Image resize — mirrors resize_base64_image(size=(1300, 600)) from notebook
# ─────────────────────────────────────────────────────────────────────────────

def _resize_b64(b64: str, size: tuple = (1300, 600)) -> str:
    """Resize a base64 JPEG to `size` and return a new base64 string."""
    try:
        img     = PILImage.open(io.BytesIO(base64.b64decode(b64)))
        resized = img.resize(size, PILImage.LANCZOS)
        buf     = io.BytesIO()
        resized.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return b64   # fall back to original if resize fails


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a helpful assistant. "
    "You will be given mixed information from documents (text, tables, and/or images). "
    "Use this information to answer the user's question accurately. "
    "Only use the provided content — do not add outside knowledge. "
    "If the content is insufficient, say so clearly. "
    "Mention the source document when relevant."
)


# ─────────────────────────────────────────────────────────────────────────────
# ChatEngine
# ─────────────────────────────────────────────────────────────────────────────

class ChatEngine:
    """Multimodal RAG engine — retrieves raw elements, answers with Gemini."""

    def __init__(self, store: MultiVectorStore):
        self._store  = store
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        print(f" ChatEngine ready — {GEMINI_MODEL}")

    # ── Public ────────────────────────────────────────────────────────────────

    def ask(self, question: str, k: int = TOP_K) -> Dict:
        """
        Retrieve top-k raw elements and ask Gemini to answer the question.

        Returns
        -------
        {
          "answer":  str,
          "sources": [{"source": str, "kind": str, "preview": str}],
          "error":   str | None
        }
        """
        try:
            elements = self._store.search(question, k=k)

            if not elements:
                return _resp(
                    "No relevant content found. Please upload and index documents first.",
                    sources=[], error=None
                )

            contents = self._build_contents(elements, question)
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
            )

            sources = [
                {
                    "source":  el.source,
                    "kind":    el.kind,
                    "preview": (
                        f"[image — base64, {len(el.content)} chars]"
                        if el.kind == "image"
                        else el.content[:300] + ("…" if len(el.content) > 300 else "")
                    ),
                }
                for el in elements
            ]

            return _resp(response.text, sources=sources, error=None)

        except Exception as exc:
            return _resp(f"Error: {exc}", sources=[], error=str(exc))

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_contents(self, elements: List[RawElement], question: str) -> list:
        """
        Build the multimodal contents list for the new google-genai SDK.

        Mirrors img_prompt_func() from the original notebook:
          - Images first, passed as inline image parts (resized to 1300×600)
          - Text and tables follow as a single formatted string
          - Question at the end
        """
        parts = [_SYSTEM, "\n\n=== DOCUMENT CONTENT ===\n"]

        text_table_parts = []
        image_parts      = []

        for i, el in enumerate(elements, 1):
            header = f"\n[{i}] Source: {el.source}  |  Type: {el.kind}\n"

            if el.kind == "image":
                image_parts.append((header, el.content))
            else:
                text_table_parts.append(header + el.content)

        # Images first — resize then pass as typed Part objects
        for header, b64 in image_parts:
            resized_b64 = _resize_b64(b64, size=(1300, 600))
            parts.append(header)
            parts.append(
                types.Part.from_bytes(
                    data=base64.b64decode(resized_b64),
                    mime_type="image/jpeg",
                )
            )

        # Then text and tables
        if text_table_parts:
            parts.append("\nText and / or tables:\n")
            parts.append("\n\n".join(text_table_parts))

        parts.append(f"\n\n=== QUESTION ===\n{question}\n\n=== ANSWER ===\n")
        return parts


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _resp(answer: str, sources: list, error) -> Dict:
    return {"answer": answer, "sources": sources, "error": error}