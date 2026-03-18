"""
Vector store service — thin wrapper around ChromaDB.

Uses cosine similarity. Single shared collection defined by
config.CHROMA_COLLECTION_NAME, persisted to config.CHROMA_PERSIST_DIR.

ChromaDB is the single source of truth for embeddings —
do NOT store them anywhere else.
"""

import hashlib
from typing import Optional

import chromadb
from loguru import logger

import backend.config as config


class VectorStoreUnavailableError(RuntimeError):
    """Raised when ChromaDB is not available (init failed or disk issue)."""


# ---------------------------------------------------------------------------
# Module-level init — deferred error so the server starts even if ChromaDB
# is broken; individual requests get a 503 instead of an import crash.
# ---------------------------------------------------------------------------

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None
_INIT_ERROR: str = ""

try:
    _client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    _collection = _client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"ChromaDB initialised — collection '{config.CHROMA_COLLECTION_NAME}'")
except Exception as _exc:
    _INIT_ERROR = str(_exc)
    logger.error(f"ChromaDB failed to initialise: {_exc}")


def _require_collection() -> chromadb.Collection:
    """Return the collection or raise VectorStoreUnavailableError."""
    if _collection is None:
        raise VectorStoreUnavailableError(
            _INIT_ERROR or "ChromaDB collection is not available."
        )
    return _collection


def collection_is_empty() -> bool:
    """Return True if no documents have been indexed yet."""
    col = _require_collection()
    return col.count() == 0


def _chunk_id(file_hash: str, chunk_index: int) -> str:
    """Return a stable deduplication ID derived from file content hash + position."""
    raw = f"{file_hash}:{chunk_index}".encode()
    return hashlib.sha256(raw).hexdigest()


def is_file_indexed(file_hash: str) -> bool:
    """Return True if any chunk from this file (by SHA-256) is already in the collection."""
    col = _require_collection()
    results = col.get(where={"file_hash": file_hash}, limit=1, include=[])
    return len(results["ids"]) > 0


def add_chunks(chunks: list[dict]) -> None:
    """
    Upsert a list of embedded chunk dicts into ChromaDB.

    Each chunk must have: text, source, file_hash, page, chunk_index, embedding.
    Chunks whose deduplication ID already exists are silently skipped.
    """
    col = _require_collection()

    if not chunks:
        logger.info("add_chunks called with empty list — nothing to do.")
        return

    ids = [_chunk_id(c["file_hash"], c["chunk_index"]) for c in chunks]

    # Determine which IDs are new
    existing = set(col.get(ids=ids, include=[])["ids"])
    new_indices = [i for i, id_ in enumerate(ids) if id_ not in existing]

    skipped = len(chunks) - len(new_indices)
    if not new_indices:
        logger.info(f"add_chunks: all {skipped} chunk(s) already indexed — skipping.")
        return

    new_chunks = [chunks[i] for i in new_indices]
    col.add(
        ids=[ids[i] for i in new_indices],
        embeddings=[c["embedding"] for c in new_chunks],
        documents=[c["text"] for c in new_chunks],
        metadatas=[
            {
                "source": c["source"],
                "file_hash": c["file_hash"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "modality": c.get("modality", ""),
                "type": c.get("type", ""),
                "gcs_uri": c.get("gcs_uri", ""),
                "section_heading": c.get("section_heading", ""),
                "document_title": c.get("document_title", ""),
            }
            for c in new_chunks
        ],
    )
    logger.info(f"add_chunks: added {len(new_indices)}, skipped {skipped} duplicate(s).")


def search(query_embedding: list[float], top_k: int = config.VECTOR_TOP_K) -> list[dict]:
    """
    Return the top_k most similar chunks for a given query embedding.

    Each result dict has: text, source, page, chunk_index, score.
    Score is cosine similarity (1 = identical, 0 = orthogonal).
    """
    col = _require_collection()
    logger.debug(f"Vector search top_k={top_k}")
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "chunk_index": meta["chunk_index"],
                "type": meta.get("type", ""),
                "modality": meta.get("modality", ""),
                "gcs_uri": meta.get("gcs_uri", ""),
                # ChromaDB returns L2-normalised cosine distance in [0, 2];
                # convert to similarity: score = 1 - distance
                "score": 1.0 - dist,
            }
        )
    return hits
