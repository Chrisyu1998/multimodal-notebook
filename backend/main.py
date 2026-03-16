"""
FastAPI application entry point.
Registers CORS middleware, mounts routers, and logs config on startup.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

import backend.config as config
from backend.routers import upload, query

# TODO (Week 3): from backend.routers import eval


# ---------------------------------------------------------------------------
# Lifespan — runs on startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    _log_config()

    # TODO: from backend.services import bm25_index
    # TODO: bm25_index.load_index()   — restore persisted BM25 index

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
# TODO (Week 3): app.include_router(eval.router, prefix="/eval", tags=["eval"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Liveness probe — returns status and current environment."""
    return {"status": "ok", "environment": config.ENVIRONMENT}


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
    logger.info(
        f"  CONTEXT_WINDOW_THRESHOLD : {config.CONTEXT_WINDOW_FILL_THRESHOLD}"
    )
    # Eval
    logger.info(f"  EVAL_DB_PATH             : {config.EVAL_DB_PATH}")
    logger.info(f"  EVAL_DATASET_PATH        : {config.EVAL_DATASET_PATH}")
    logger.info(f"  EVAL_JUDGE_MODEL         : {config.EVAL_JUDGE_MODEL}")
    logger.info("============================")
