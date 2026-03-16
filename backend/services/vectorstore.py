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


def _chunk_id(file_hash: str, chunk_index: int) -> str:
    """Return a stable deduplication ID derived from file content hash + position."""
    raw = f"{file_hash}:{chunk_index}".encode()
    return hashlib.sha256(raw).hexdigest()


def is_file_indexed(file_hash: str) -> bool:
    """Return True if any chunk from this file (by SHA-256) is already in the collection."""
    results = _collection.get(where={"file_hash": file_hash}, limit=1, include=[])
    return len(results["ids"]) > 0


def add_chunks(chunks: list[dict]) -> None:
    """
    Upsert a list of embedded chunk dicts into ChromaDB.

    Each chunk must have: text, source, file_hash, page, chunk_index, embedding.
    Chunks whose deduplication ID already exists are silently skipped.
    """
    if not chunks:
        logger.info("add_chunks called with empty list — nothing to do.")
        return

    ids = [_chunk_id(c["file_hash"], c["chunk_index"]) for c in chunks]

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
            {
                "source": c["source"],
                "file_hash": c["file_hash"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "modality": c.get("modality", ""),
                "type": c.get("type", ""),
            }
            for c in new_chunks
        ],
    )
    logger.info(f"add_chunks: added {len(new_indices)}, skipped {skipped} duplicate(s).")


_KNOWN_TYPES: list[str] = ["document", "video", "audio", "image"]


def search_balanced(
    query_embedding: list[float],
    global_top_k: int = 20,
    top_k_per_modality: int = 5,
) -> list[dict]:
    """Return a modality-balanced chunk pool suitable for reranking.

    Step 1: Run a global top-k search.
    Step 2: Check which content types (document, video, audio, image) are absent.
    Step 3: For each absent type that exists in the collection, inject up to
            top_k_per_modality results via a filtered search.
    Step 4: Return the merged, deduplicated list sorted by score descending.
    """
    global_hits = search(query_embedding, top_k=global_top_k)

    represented = {h.get("type") for h in global_hits if h.get("type")}
    missing_types = [t for t in _KNOWN_TYPES if t not in represented]

    if not missing_types:
        return global_hits

    seen_ids: set[str] = {
        f"{h['source']}:{h['chunk_index']}" for h in global_hits
    }
    supplemental: list[dict] = []

    for content_type in missing_types:
        try:
            results = _collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k_per_modality,
                where={"type": content_type},
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # Collection may have no chunks of this type — skip silently
            continue

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            uid = f"{meta['source']}:{meta['chunk_index']}"
            if uid in seen_ids:
                continue
            seen_ids.add(uid)
            supplemental.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "page": meta["page"],
                    "chunk_index": meta["chunk_index"],
                    "type": meta.get("type", ""),
                    "score": 1.0 - dist,
                }
            )

    if supplemental:
        logger.info(
            f"search_balanced: injected {len(supplemental)} chunk(s) from "
            f"missing modalities {missing_types}"
        )

    merged = global_hits + supplemental
    merged.sort(key=lambda h: h["score"], reverse=True)
    return merged


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
                "type": meta.get("type", ""),
                # ChromaDB returns L2-normalised cosine distance in [0, 2];
                # convert to similarity: score = 1 - distance
                "score": 1.0 - dist,
            }
        )
    return hits
