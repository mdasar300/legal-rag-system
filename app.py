"""
app.py — Streamlit UI for the Multimodal RAG system.

Full pipeline per uploaded document:
  1. extractor.py   → text_chunks, tables, images (base64)
  2. summariser.py  → table_summaries, image_summaries   (text only, via Gemini)
  3. store.py       → text chunks embedded directly (HuggingFace)
                      table summaries embedded, raw tables stored
                      image summaries embedded, raw base64 stored
  4. [User asks]
  5. store.search() → top-k raw elements (text / table / image)
  6. chat_engine.py → multimodal Gemini prompt → answer

Run with:  streamlit run app.py
"""

import shutil
from pathlib import Path

import streamlit as st
from google import genai

from config import UPLOAD_DIR, SUPPORTED_EXTENSIONS, MAX_FILE_MB, TOP_K, GEMINI_API_KEY, GEMINI_MODEL
from extractor   import extract
from summariser  import summarise_tables, summarise_images
from store       import MultiVectorStore
from chat_engine import ChatEngine

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Document RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .title { font-size:2rem; font-weight:800; color:#1a3c6e; }
    .sub   { font-size:0.9rem; color:#666; margin-bottom:1.5rem; }
    .card  {
        border:1px solid #dde; border-radius:8px;
        padding:0.7rem 1rem; margin-bottom:0.5rem;
        background:#f9f9fc; font-size:0.86rem;
    }
    .badge {
        display:inline-block; padding:1px 8px; border-radius:12px;
        font-size:0.74rem; font-weight:600; margin-left:6px;
    }
    .text  { background:#dbeafe; color:#1d4ed8; }
    .table { background:#dcfce7; color:#15803d; }
    .image { background:#fef9c3; color:#a16207; }
    .mode-rag { background:#ede9fe; color:#6d28d9; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .mode-llm { background:#fee2e2; color:#b91c1c; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    for key, val in [("store", None), ("engine", None),
                     ("history", []), ("indexed", []),
                     ("mode", "RAG")]:           # default mode = RAG
        st.session_state.setdefault(key, val)

_init()


# ── Shared instances ──────────────────────────────────────────────────────────

def get_store() -> MultiVectorStore:
    if st.session_state.store is None:
        store = MultiVectorStore()
        store.load()
        st.session_state.store = store
    return st.session_state.store


def get_engine() -> ChatEngine:
    if st.session_state.engine is None:
        st.session_state.engine = ChatEngine(get_store())
    return st.session_state.engine


# ── LLM-only call (no retrieval) ──────────────────────────────────────────────

def ask_llm_only(question: str) -> dict:
    """
    Call Gemini directly — no document retrieval at all.
    Used when the user switches to LLM mode to ask general questions.
    """
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                "You are a helpful general-purpose assistant. "
                "Answer the user's question using your own knowledge.",
                f"\nQuestion: {question}\nAnswer:",
            ],
        )
        return {"answer": response.text, "sources": [], "error": None, "mode": "LLM"}
    except Exception as exc:
        return {"answer": f"Error: {exc}", "sources": [], "error": str(exc), "mode": "LLM"}


# ── Indexing pipeline ─────────────────────────────────────────────────────────

def index_files(uploaded_files):
    store   = get_store()
    bar     = st.progress(0, text="Starting…")
    success = 0

    for idx, uf in enumerate(uploaded_files):
        ext     = Path(uf.name).suffix.lower()
        size_mb = uf.size / 1024 / 1024
        pct     = idx / len(uploaded_files)

        if ext not in SUPPORTED_EXTENSIONS:
            st.error(f"❌ {uf.name}: unsupported format.")
            continue
        if size_mb > MAX_FILE_MB:
            st.error(f"❌ {uf.name}: too large ({size_mb:.1f} MB, max {MAX_FILE_MB} MB).")
            continue

        bar.progress(pct, text=f"Processing {uf.name}…")
        dest = UPLOAD_DIR / uf.name
        dest.write_bytes(uf.getbuffer())

        try:
            # ── Step 1: Extract ───────────────────────────────────────────
            with st.spinner(f"Extracting from {uf.name}…"):
                text_chunks, tables, images = extract(dest)

            st.write(
                f"📄 **{uf.name}** — "
                f"{len(text_chunks)} text chunks · "
                f"{len(tables)} tables · "
                f"{len(images)} images"
            )

            # ── Step 2: Summarise tables and images (NOT text) ────────────
            table_summaries = []
            image_summaries = []

            if tables:
                with st.spinner(f"Summarising {len(tables)} table(s)…"):
                    table_summaries = summarise_tables(tables)

            if images:
                with st.spinner(f"Summarising {len(images)} image(s)…"):
                    image_summaries = summarise_images(images)

            # ── Step 3: Add to store ──────────────────────────────────────
            if text_chunks:
                store.add_texts(text_chunks, source=uf.name)
            if tables:
                store.add_tables(table_summaries, tables, source=uf.name)
            if images:
                store.add_images(image_summaries, images, source=uf.name)

            if uf.name not in st.session_state.indexed:
                st.session_state.indexed.append(uf.name)

            success += 1

        except Exception as exc:
            st.error(f"❌ {uf.name}: {exc}")
            continue

    store.save()
    st.session_state.engine = ChatEngine(store)

    bar.progress(1.0, text="Done!")
    if success:
        st.success(
            f"✅ Indexed {success} file(s). "
            f"Total vectors in store: {store.total}"
        )


def clear_all():
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for f in (MultiVectorStore._INDEX_FILE, MultiVectorStore._DATA_FILE):
        f.unlink(missing_ok=True)
    st.session_state.update(store=None, engine=None, history=[], indexed=[], mode="RAG")
    st.success("All data cleared.")
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Documents")

    uploaded = st.file_uploader(
        "Upload files",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )

    st.markdown("---")
    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, TOP_K)

    st.markdown("---")

    if st.button("🔄 Extract, Summarise & Index", use_container_width=True):
        if uploaded:
            index_files(uploaded)
        else:
            st.warning("Upload at least one document first.")

    if st.button("🗑️ Clear Everything", use_container_width=True):
        clear_all()

    store = get_store()
    if store.total > 0:
        st.markdown("---")
        st.metric("Vectors indexed", store.total)
        st.metric("Files processed", len(st.session_state.indexed))
        for fname in st.session_state.indexed:
            st.caption(f"✅ {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="title">🔍 Multimodal Document RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">'
    'Text embedded directly · Tables & images summarised first · '
    'Raw originals retrieved · Gemini answers with full multimodal context'
    '</div>',
    unsafe_allow_html=True,
)

tab_chat, tab_docs, tab_about = st.tabs(["💬 Chat", "📄 Indexed Files", "ℹ️ About"])


# ── Chat ──────────────────────────────────────────────────────────────────────
with tab_chat:

    no_docs = get_store().total == 0

    if no_docs and st.session_state.mode == "RAG":
        st.info("👈 Upload and index documents using the sidebar, or switch to **LLM** mode to chat freely.")

    # ── Question input ────────────────────────────────────────────────────────
    question = st.text_input(
        "Ask a question",
        placeholder=(
            "e.g. Summarise the figures in this paper."
            if st.session_state.mode == "RAG"
            else "e.g. What is the capital of France?"
        ),
    )

    # ── RAG / LLM toggle + Ask + Clear — all in one row below the input ───────
    mode_col1, mode_col2, gap_col, ask_col, clear_col = st.columns([1, 1, 2, 1, 1])

    with mode_col1:
        if st.button(
            "🗂️ RAG" + (" ✓" if st.session_state.mode == "RAG" else ""),
            use_container_width=True,
            type="primary" if st.session_state.mode == "RAG" else "secondary",
        ):
            st.session_state.mode = "RAG"
            st.rerun()

    with mode_col2:
        if st.button(
            "🤖 LLM" + (" ✓" if st.session_state.mode == "LLM" else ""),
            use_container_width=True,
            type="primary" if st.session_state.mode == "LLM" else "secondary",
        ):
            st.session_state.mode = "LLM"
            st.rerun()

    with ask_col:
        ask_btn = st.button("Ask ➜", type="primary", use_container_width=True)

    with clear_col:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    # ── Mode hint ─────────────────────────────────────────────────────────────
    if st.session_state.mode == "RAG":
        st.caption("🗂️ **RAG mode** — answers grounded in your indexed documents.")
    else:
        st.caption("🤖 **LLM mode** — answers from Gemini's general knowledge, no documents used.")

    # ── Handle Ask ────────────────────────────────────────────────────────────
    if ask_btn and question.strip():
        if st.session_state.mode == "RAG":
            if no_docs:
                st.warning("No documents indexed yet. Switch to LLM mode or upload documents first.")
            else:
                with st.spinner("Retrieving content and generating answer…"):
                    result = get_engine().ask(question, k=top_k)
                result["mode"] = "RAG"
                st.session_state.history.append({"q": question, **result})
        else:
            with st.spinner("Asking Gemini directly…"):
                result = ask_llm_only(question)
            st.session_state.history.append({"q": question, **result})

    # ── Render conversation — newest first ────────────────────────────────────
    for item in reversed(st.session_state.history):
        mode_badge = (
            '<span class="mode-rag">🗂️ RAG</span>'
            if item.get("mode") == "RAG"
            else '<span class="mode-llm">🤖 LLM</span>'
        )
        st.markdown(
            f"**🙋 {item['q']}** &nbsp; {mode_badge}",
            unsafe_allow_html=True,
        )

        if item["error"]:
            st.error(item["answer"])
        else:
            st.markdown(item["answer"])

            if item.get("sources"):
                with st.expander(f"📚 Retrieved elements ({len(item['sources'])})"):
                    for s in item["sources"]:
                        st.markdown(
                            f'<div class="card">'
                            f'📄 <strong>{s["source"]}</strong>'
                            f'<span class="badge {s["kind"]}">{s["kind"]}</span>'
                            f'<br><small>{s["preview"]}</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
        st.divider()


# ── Indexed files ─────────────────────────────────────────────────────────────
with tab_docs:
    if not st.session_state.indexed:
        st.info("No documents indexed yet.")
    else:
        st.write(f"**{len(st.session_state.indexed)} file(s) indexed:**")
        for fname in st.session_state.indexed:
            st.markdown(f"- ✅ {fname}")


# ── About ─────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
### How the pipeline works

```
Upload document
      │
      ▼
 extractor.py
      ├── text chunks     (word-based sliding window)
      ├── table strings   (one raw string per table)
      └── base64 images   (one JPEG per embedded image)
      │
      ▼
 summariser.py  ← Gemini
      ├── text chunks  →  NO summary — embedded directly ✓
      ├── table strings → summary for embedding, raw table stored
      └── base64 images → summary for embedding, raw image stored
      │
      ▼
 store.py
      ├── embed with HuggingFace all-MiniLM-L6-v2
      ├── store vectors in FAISS (cosine similarity)
      └── store raw originals in UUID-keyed dict
      │
      ▼
 [User asks question]
      │
      ├── RAG mode → store.search() → raw elements → Gemini (grounded)
      └── LLM mode → Gemini directly (general knowledge, no docs)
```

| Component | Technology |
|---|---|
| Summarisation + final Q&A | Google Gemini 2.5 Flash |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS (cosine similarity) |
| PDF parsing | `unstructured` hi_res |
| DOCX / PPTX | `python-docx` / `python-pptx` |
| UI | Streamlit |

**Supported formats:** PDF · DOCX · DOC · PPTX · PPT · TXT
""")