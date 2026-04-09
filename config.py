"""
config.py — All settings in one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API ───────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
print("API KEY:", GEMINI_API_KEY)

# Correct model name for the new google-genai SDK
# Use gemini-2.0-flash — fastest, cheapest, fully multimodal, latest generation
# Alternatives: "gemini-1.5-pro", "gemini-2.0-pro-exp"
GEMINI_MODEL = "gemini-2.5-flash"

# ── Embeddings ────────────────────────────────────────────────────────────────
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM      = 384

# ── Chunking (for raw text only) ──────────────────────────────────────────────
CHUNK_SIZE    = 800   # words per chunk
CHUNK_OVERLAP = 100   # word overlap between consecutive chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
UPLOAD_DIR  = DATA_DIR / "uploads"
EXTRACT_DIR = DATA_DIR / "extracted"
STORE_DIR   = DATA_DIR / "store"

for _d in [UPLOAD_DIR, EXTRACT_DIR, STORE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Supported formats ─────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt"}
MAX_FILE_MB = 50