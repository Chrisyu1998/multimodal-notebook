"""
Vector store service — thin wrapper around ChromaDB.

Uses cosine similarity. Single shared collection defined by
config.CHROMA_COLLECTION_NAME, persisted to config.CHROMA_PERSIST_DIR.

ChromaDB is the single source of truth for embeddings —
do NOT store them anywhere else.
"""

import hashlib

import chromadb
from loguru import logger

import backend.config as config

_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
_collection = _client.get_or_create_collection(
    name=config.CHROMA_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def _chunk_id(source: str, chunk_index: int) -> str:
    """Return a stable SHA-256 deduplication ID for a chunk."""
    raw = f"{source}:{chunk_index}".encode()
    return hashlib.sha256(raw).hexdigest()


def add_chunks(chunks: list[dict]) -> None:
    """
    Upsert a list of embedded chunk dicts into ChromaDB.

    Each chunk must have: text, source, page, chunk_index, embedding.
    Chunks whose deduplication ID already exists are silently skipped.
    """
    if not chunks:
        logger.info("add_chunks called with empty list — nothing to do.")
        return

    ids = [_chunk_id(c["source"], c["chunk_index"]) for c in chunks]

    # Determine which IDs are new
    existing = set(_collection.get(ids=ids, include=[])["ids"])
    new_indices = [i for i, id_ in enumerate(ids) if id_ not in existing]

    skipped = len(chunks) - len(new_indices)
    if not new_indices:
        logger.info(f"add_chunks: all {skipped} chunk(s) already indexed — skipping.")
        return

    new_chunks = [chunks[i] for i in new_indices]
    _collection.add(
        ids=[ids[i] for i in new_indices],
        embeddings=[c["embedding"] for c in new_chunks],
        documents=[c["text"] for c in new_chunks],
        metadatas=[
            {"source": c["source"], "page": c["page"], "chunk_index": c["chunk_index"]}
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
    logger.debug(f"Vector search top_k={top_k}")
    results = _collection.query(
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
                # ChromaDB returns L2-normalised cosine distance in [0, 2];
                # convert to similarity: score = 1 - distance
                "score": 1.0 - dist,
            }
        )
    return hits
