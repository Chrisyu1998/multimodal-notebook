"""
FastAPI application entry point.
Registers CORS middleware, mounts routers, and logs config on startup.
"""

import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

import backend.config as config
from backend.routers import upload, query, eval as eval_router, metrics as metrics_router
from backend.services import bm25_index
from backend.services.vectorstore import get_doc_count, VectorStoreUnavailableError


# ---------------------------------------------------------------------------
# Lifespan — runs on startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    _log_config()
    bm25_index.load_index()

    # TODO: ping ChromaDB to verify the persistence dir is accessible
    #   import chromadb
    #   chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

    logger.info("API ready.")
    yield
    # ---- shutdown ----
    logger.info("API shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multimodal RAG API",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(eval_router.router, prefix="/eval", tags=["eval"])
app.include_router(metrics_router.router, prefix="/metrics", tags=["metrics"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """System status probe — returns environment, ChromaDB doc count, BM25 index size,
    SQLite eval run count, and GCS bucket name."""
    # ChromaDB
    chroma_status = "ok"
    chroma_doc_count = 0
    try:
        chroma_doc_count = get_doc_count()
    except VectorStoreUnavailableError as exc:
        chroma_status = f"unavailable: {exc}"

    # BM25
    bm25_size = bm25_index.get_index_size()

    # SQLite eval run count
    eval_run_count = 0
    try:
        conn = sqlite3.connect(config.EVAL_DB_PATH)
        try:
            row = conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()
            eval_run_count = row[0] if row else 0
        finally:
            conn.close()
    except Exception:
        pass  # DB may not exist yet on a fresh install

    return {
        "status": "ok",
        "environment": config.ENVIRONMENT,
        "chroma": {
            "status": chroma_status,
            "doc_count": chroma_doc_count,
            "collection": config.CHROMA_COLLECTION_NAME,
        },
        "bm25": {
            "index_size": bm25_size,
        },
        "eval": {
            "run_count": eval_run_count,
            "db_path": config.EVAL_DB_PATH,
        },
        "gcs": {
            "bucket": config.GCS_BUCKET_NAME,
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mask(key: str) -> str:
    """Show only the last 4 characters of a secret key."""
    return f"{'*' * (len(key) - 4)}{key[-4:]}" if len(key) > 4 else "****"


def _log_config() -> None:
    """Log all loaded config values at startup (secrets masked)."""
    logger.info("=== Loaded configuration ===")
    logger.info(f"  ENVIRONMENT              : {config.ENVIRONMENT}")
    logger.info(f"  HOST:PORT                : {config.HOST}:{config.PORT}")
    # Gemini
    logger.info(f"  GEMINI_API_KEY           : {_mask(config.GEMINI_API_KEY)}")
    logger.info(f"  EMBEDDING_MODEL          : {config.EMBEDDING_MODEL}")
    logger.info(f"  GENERATION_MODEL         : {config.GENERATION_MODEL}")
    logger.info(f"  RERANK_MODEL             : {config.RERANK_MODEL}")
    # GCS
    logger.info(f"  GCS_BUCKET_NAME          : {config.GCS_BUCKET_NAME}")
    # ChromaDB
    logger.info(f"  CHROMA_PERSIST_DIR       : {config.CHROMA_PERSIST_DIR}")
    logger.info(f"  CHROMA_COLLECTION_NAME   : {config.CHROMA_COLLECTION_NAME}")
    # Chunking
    logger.info(f"  TMP_UPLOAD_DIR           : {config.TMP_UPLOAD_DIR}")
    # Retrieval
    logger.info(f"  BM25_TOP_K               : {config.BM25_TOP_K}")
    logger.info(f"  VECTOR_TOP_K             : {config.VECTOR_TOP_K}")
    logger.info(f"  RERANK_TOP_K             : {config.RERANK_TOP_K}")
    # Eval
    logger.info(f"  EVAL_DB_PATH             : {config.EVAL_DB_PATH}")
    logger.info(f"  EVAL_DATASET_PATH        : {config.EVAL_DATASET_PATH}")
    logger.info(f"  EVAL_JUDGE_MODEL         : {config.EVAL_JUDGE_MODEL}")
    logger.info("============================")
