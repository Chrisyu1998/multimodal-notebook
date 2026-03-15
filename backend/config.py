"""
Central configuration module.

Loads .env at import time via python-dotenv. Every other module must
import constants from here — no file may call os.environ or load_dotenv
directly. All variables are typed and grouped to match .env.example.

Raises ValueError at startup if any required variable is absent.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve .env relative to this file so the server can be started
# from any working directory.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


def _require(name: str) -> str:
    """Return the value of *name* from the environment or raise ValueError."""
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Required environment variable '{name}' is missing or empty. "
            f"Copy .env.example to .env and set a value for it."
        )
    return value


def _optional(name: str, default: str) -> str:
    return os.getenv(name, default)


def _optional_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _optional_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


# ============================================================
# Gemini / Google AI
# ============================================================

GEMINI_API_KEY: str = _require("GEMINI_API_KEY")

EMBEDDING_MODEL: str = "gemini-embedding-2-preview"  # natively multimodal
EMBEDDING_BATCH_SIZE: int = _optional_int("EMBEDDING_BATCH_SIZE", 20)
EMBEDDING_IMAGE_BATCH_SIZE: int = _optional_int("EMBEDDING_IMAGE_BATCH_SIZE", 6)        # hard API limit
EMBEDDING_DOCUMENT_BATCH_SIZE: int = _optional_int("EMBEDDING_DOCUMENT_BATCH_SIZE", 20)   # 20,000 token limit / ~800 tokens per PDF chunk ≈ 25 max; 20 is a safe default
EMBEDDING_MAX_RETRIES: int = _optional_int("EMBEDDING_MAX_RETRIES", 3)
EMBEDDING_MAX_WORKERS: int = _optional_int("EMBEDDING_MAX_WORKERS", 8)
EMBEDDING_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
GENERATION_MODEL: str = "gemini-1.5-pro"        # immutable — see CLAUDE.md
RERANK_MODEL: str = "gemini-1.5-flash"          # immutable — see CLAUDE.md

# ============================================================
# Google Cloud Storage
# ============================================================

GCS_BUCKET_NAME: str = _require("GCS_BUCKET_NAME")

# ============================================================
# ChromaDB
# ============================================================

CHROMA_PERSIST_DIR: str = _optional("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME: str = _optional("CHROMA_COLLECTION_NAME", "rag_chunks")

# ============================================================
# FastAPI
# ============================================================

ENVIRONMENT: str = _optional("ENVIRONMENT", "development")
HOST: str = _optional("HOST", "127.0.0.1")
PORT: int = _optional_int("PORT", 8000)

# ============================================================
# Upload validation
# ============================================================

ALLOWED_FILE_TYPES: set[str] = {".pdf", ".png", ".jpeg", ".mp4", ".mov"}
MAX_FILE_SIZE_MB: int = _optional_int("MAX_FILE_SIZE_MB", 500)

# ============================================================
# Chunking
# ============================================================

CHUNK_SIZE: int = _optional_int("CHUNK_SIZE", 512)
CHUNK_OVERLAP: int = _optional_int("CHUNK_OVERLAP", 64)
PDF_CHUNK_SIZE: int = _optional_int("PDF_CHUNK_SIZE", 800)
PDF_CHUNK_OVERLAP: int = _optional_int("PDF_CHUNK_OVERLAP", 100)
VIDEO_MAX_SCENE_DURATION: float = _optional_float("VIDEO_MAX_SCENE_DURATION", 120.0)  # Gemini API hard limit (seconds)
VIDEO_FORCED_SPLIT_OVERLAP: float = _optional_float("VIDEO_FORCED_SPLIT_OVERLAP", 5.0)  # overlap between forced sub-segments
VIDEO_SUMMARY_MAX_TOKENS: int = _optional_int("VIDEO_SUMMARY_MAX_TOKENS", 2048)  # gemini-embedding-2-preview hard limit is 8192; 2048 is enough for a 128s clip
TMP_UPLOAD_DIR: str = _optional("TMP_UPLOAD_DIR", "./tmp")
TMP_UPLOADS_DIR: str = str(Path(_optional("TMP_UPLOAD_DIR", "./tmp")) / "uploads")

# ============================================================
# Retrieval
# ============================================================

BM25_TOP_K: int = _optional_int("BM25_TOP_K", 20)
VECTOR_TOP_K: int = _optional_int("VECTOR_TOP_K", 20)
RERANK_TOP_K: int = _optional_int("RERANK_TOP_K", 5)
CONTEXT_WINDOW_FILL_THRESHOLD: float = _optional_float(
    "CONTEXT_WINDOW_FILL_THRESHOLD", 0.80
)

# ============================================================
# Eval
# ============================================================

EVAL_DB_PATH: str = _optional(
    "EVAL_DB_PATH", "./backend/eval/eval_results.db"
)
EVAL_DATASET_PATH: str = _optional(
    "EVAL_DATASET_PATH", "./backend/eval/golden_dataset.json"
)
EVAL_JUDGE_MODEL: str = _optional("EVAL_JUDGE_MODEL", "gemini-1.5-pro")
