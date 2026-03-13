"""
BM25 keyword search index — wraps rank-bm25.

Important: the index is built during ingestion (never in the query path).
On startup, main.py calls load_index() to restore the persisted index.
"""

import pickle
from pathlib import Path
from loguru import logger

import backend.config as config

# TODO: from rank_bm25 import BM25Okapi

_BM25_INDEX_PATH = Path(config.CHROMA_PERSIST_DIR) / "bm25_index.pkl"

# Module-level index instance — populated by build_index() or load_index()
_index = None
_corpus_chunks: list[dict] = []   # parallel list to _index corpus


def build_index(chunks: list[dict]) -> None:
    """
    Build a BM25 index from the text of all chunks and persist it to disk.
    Called at the end of every ingestion run — replaces the previous index.
    """
    global _index, _corpus_chunks
    logger.info(f"Building BM25 index over {len(chunks)} chunks…")
    # TODO: tokenize each chunk's "text" field (simple whitespace split is fine)
    # TODO: instantiate BM25Okapi(tokenized_corpus)
    # TODO: persist (_index, _corpus_chunks) to _BM25_INDEX_PATH with pickle
    raise NotImplementedError


def load_index() -> None:
    """
    Load a previously persisted BM25 index from disk into memory.
    Called once at application startup by main.py.
    No-op if the index file does not exist yet.
    """
    global _index, _corpus_chunks
    if not _BM25_INDEX_PATH.exists():
        logger.warning("No BM25 index found on disk — will be created after first upload.")
        return
    # TODO: unpickle (_index, _corpus_chunks) from _BM25_INDEX_PATH
    logger.info("BM25 index loaded from disk.")


def search_bm25(query: str, top_k: int = 20) -> list[dict]:
    """
    Return the top_k chunks most relevant to the query string via BM25.
    Raises RuntimeError if the index has not been built yet.
    """
    if _index is None:
        raise RuntimeError("BM25 index is not loaded. Run an ingestion first.")
    logger.debug(f"BM25 search top_k={top_k}: {query!r}")
    # TODO: tokenize query (same method as build_index)
    # TODO: call _index.get_top_n(tokenized_query, _corpus_chunks, n=top_k)
    # TODO: return list of chunk dicts (scores attached as "score" field)
    raise NotImplementedError
