"""
Vector store service — thin wrapper around ChromaDB.

Uses cosine similarity. Single shared collection defined by
config.CHROMA_COLLECTION_NAME, persisted to config.CHROMA_PERSIST_DIR.

ChromaDB is the single source of truth for embeddings —
do NOT store them anywhere else.
"""

from loguru import logger

import backend.config as config

# TODO: import chromadb and initialize persistent client at module level
#   _client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
#   _collection = _client.get_or_create_collection(
#       name=config.CHROMA_COLLECTION_NAME,
#       metadata={"hnsw:space": "cosine"},
#   )


def add_chunks(chunks: list[dict]) -> None:
    """
    Upsert a list of embedded chunk dicts into ChromaDB.
    Expects each chunk to have an "embedding" key populated by embed_chunks().
    """
    logger.info(f"Adding {len(chunks)} chunks to ChromaDB…")
    # TODO: extract ids, embeddings, documents, and metadatas from chunks
    # TODO: call _collection.upsert(ids=..., embeddings=..., documents=..., metadatas=...)


def search(query_embedding: list[float], top_k: int = 20) -> list[dict]:
    """
    Return the top_k most similar chunks for a given query embedding.
    Each result dict mirrors the chunk shape plus a "score" field.
    """
    logger.debug(f"Vector search top_k={top_k}")
    # TODO: call _collection.query(query_embeddings=[query_embedding], n_results=top_k)
    # TODO: unpack results and return list of chunk dicts with scores
    raise NotImplementedError
