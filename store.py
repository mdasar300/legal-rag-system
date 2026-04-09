"""
store.py — MultiVector store inspired by the notebook's MultiVectorRetriever.

Two layers:
  1. FAISS index     — stores HuggingFace embedding vectors of:
                         • raw text chunks   (embedded directly, no summary)
                         • table summaries   (summary embedded, raw table stored)
                         • image summaries   (summary embedded, raw b64 stored)

  2. Docstore (dict) — maps UUID → RawElement (raw content + metadata)
                       FAISS row → UUID → raw original

At query time:
  embed(query) → FAISS top-k → UUIDs → raw originals → LLM prompt
"""

import pickle
import uuid
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import HF_EMBEDDING_MODEL, EMBEDDING_DIM, TOP_K, STORE_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

class RawElement:
    """One retrieved unit — what the LLM actually sees."""
    __slots__ = ("content", "kind", "source")

    def __init__(self, content: str, kind: str, source: str):
        self.content = content  # raw text chunk / raw table string / raw base64 image
        self.kind    = kind     # "text" | "table" | "image"
        self.source  = source   # original filename


# ─────────────────────────────────────────────────────────────────────────────
# MultiVectorStore
# ─────────────────────────────────────────────────────────────────────────────

class MultiVectorStore:
    """
    Index embeddings, return raw originals.

    add_texts()   — embed text chunks directly, store raw chunks
    add_tables()  — embed table summaries, store raw table strings
    add_images()  — embed image summaries, store raw base64 strings
    search()      — embed query → FAISS search → return raw RawElements
    """

    _INDEX_FILE = STORE_DIR / "faiss.index"
    _DATA_FILE  = STORE_DIR / "docstore.pkl"

    def __init__(self):
        print(f"Loading embedding model: {HF_EMBEDDING_MODEL} …")
        self._embedder: SentenceTransformer = SentenceTransformer(HF_EMBEDDING_MODEL)
        self._index:    faiss.IndexFlatIP   = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._id_map:   List[str]           = []   # FAISS row index → UUID
        self._docstore: Dict[str, RawElement] = {} # UUID → RawElement

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self._embedder.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine via inner-product on unit vectors
        ).astype("float32")

    # ── Internal: add (embedding_texts, raw_contents) pair ────────────────────

    def _add(self, embed_texts: List[str], raw_contents: List[str], kind: str, source: str) -> None:
        """
        embed_texts  — what gets embedded into FAISS (text chunks / summaries)
        raw_contents — what gets stored in docstore (always the raw original)
        """
        if not embed_texts:
            return

        vecs = self._embed(embed_texts)
        self._index.add(vecs)

        for raw in raw_contents:
            doc_id = str(uuid.uuid4())
            self._id_map.append(doc_id)
            self._docstore[doc_id] = RawElement(content=raw, kind=kind, source=source)

    # ── Public: add each element type ─────────────────────────────────────────

    def add_texts(self, chunks: List[str], source: str) -> None:
        """Text chunks: embedded directly, stored as-is."""
        self._add(
            embed_texts=chunks,
            raw_contents=chunks,
            kind="text",
            source=source,
        )

    def add_tables(self, summaries: List[str], raw_tables: List[str], source: str) -> None:
        """Tables: summaries embedded, raw table strings stored."""
        self._add(
            embed_texts=summaries,
            raw_contents=raw_tables,
            kind="table",
            source=source,
        )

    def add_images(self, summaries: List[str], raw_b64: List[str], source: str) -> None:
        """Images: summaries embedded, raw base64 strings stored."""
        self._add(
            embed_texts=summaries,
            raw_contents=raw_b64,
            kind="image",
            source=source,
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = TOP_K) -> List[RawElement]:
        """
        Embed query, find nearest vectors in FAISS,
        return the linked raw RawElements.
        """
        if self._index.ntotal == 0:
            return []

        q_vec = self._embed([query])
        k = min(k, self._index.ntotal)
        _, indices = self._index.search(q_vec, k)

        # Be robust to any historical mismatch between FAISS rows and id_map/docstore
        # (e.g. after code changes or partially written files).
        results: List[RawElement] = []
        for i in indices[0]:
            if 0 <= i < len(self._id_map):
                doc_id = self._id_map[i]
                el = self._docstore.get(doc_id)
                if el is not None:
                    results.append(el)
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        faiss.write_index(self._index, str(self._INDEX_FILE))
        with open(self._DATA_FILE, "wb") as f:
            # Store a plain-JSON-like structure so we don't depend on the
            # exact in-memory RawElement class identity (Streamlit reloads
            # modules, which can break pickling of custom classes).
            serialisable_docstore = {
                doc_id: {
                    "content": el.content,
                    "kind": el.kind,
                    "source": el.source,
                }
                for doc_id, el in self._docstore.items()
            }
            pickle.dump(
                {"id_map": self._id_map, "docstore": serialisable_docstore},
                f,
            )
        print(f"  Saved store: {self._index.ntotal} vectors.")

    def load(self) -> bool:
        if not self._INDEX_FILE.exists():
            return False
        try:
            self._index = faiss.read_index(str(self._INDEX_FILE))
            with open(self._DATA_FILE, "rb") as f:
                data = pickle.load(f)
            self._id_map = data["id_map"]

            # Support both the new dict format and any older RawElement-based
            # pickles (best-effort backwards compatibility).
            raw_docstore = data["docstore"]
            if raw_docstore and isinstance(next(iter(raw_docstore.values())), RawElement):
                # Old format: already RawElement instances.
                self._docstore = raw_docstore
            else:
                # New format: reconstruct RawElement objects from dicts.
                self._docstore = {
                    doc_id: RawElement(
                        content=payload["content"],
                        kind=payload["kind"],
                        source=payload["source"],
                    )
                    for doc_id, payload in raw_docstore.items()
                }
            print(f"  Loaded store: {self._index.ntotal} vectors.")
            return True
        except Exception as exc:
            print(f"  Could not load store: {exc}")
            return False

    @property
    def total(self) -> int:
        return self._index.ntotal