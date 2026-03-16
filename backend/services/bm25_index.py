"""
BM25 keyword search index — wraps rank-bm25.

Important: the index is built during ingestion (never in the query path).
On startup, main.py calls load_index() to restore the persisted index.

Accumulation semantics: build_index() appends new chunks to the existing
in-memory corpus and rebuilds the BM25Okapi model so every ingested file
remains searchable across server restarts.

Tokenizer: regex word-split + Porter stemming so "running", "run", and
"runs" all map to the same term.  _PICKLE_VERSION is bumped whenever the
tokenizer or schema changes — load_index() discards stale pickles rather
than silently producing wrong results.

Modality filter: video_clip chunks are excluded from BM25.  Their text
(the Gemini visual summary) is identical to the paired video_summary
(Chunk B) which IS indexed.  Indexing both would produce duplicate results
for every video scene.
"""

import pickle
import re
from pathlib import Path
from typing import Optional

from loguru import logger
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

import backend.config as config

_BM25_INDEX_PATH = Path(config.CHROMA_PERSIST_DIR) / "bm25_index.pkl"

# Bump this whenever the tokenizer or stored schema changes.
# load_index() discards any pickle whose version does not match.
_PICKLE_VERSION: int = 2

# video_clip chunks share identical text with their paired video_summary
# (Chunk B).  Only Chunk B is indexed so each scene appears once in BM25.
_BM25_SKIP_MODALITIES: frozenset[str] = frozenset({"video_clip"})

# Fields that hold large binary blobs — stripped before storing in the corpus
# so the pickle stays small.
_BINARY_FIELDS: frozenset[str] = frozenset(
    {"video_bytes", "audio_bytes", "pdf_bytes", "image_bytes", "embedding"}
)

# Module-level state — populated by build_index() or load_index()
_index: Optional[BM25Okapi] = None
_corpus_chunks: list[dict] = []

_stemmer = PorterStemmer()
_token_re = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> list[str]:
    """Regex word-split + Porter stem.

    Handles punctuation, numbers, and contractions without requiring any
    NLTK corpus downloads.  Applied identically at index build time and
    query time so the token spaces always match.
    """
    return [_stemmer.stem(t) for t in _token_re.findall(text.lower())]


def _strip_binaries(chunk: dict) -> dict:
    """Return a shallow copy of *chunk* with binary blob fields removed."""
    return {k: v for k, v in chunk.items() if k not in _BINARY_FIELDS}


def build_index(chunks: list[dict]) -> None:
    """Append *chunks* to the existing corpus, rebuild the BM25 model, and persist.

    Called at the end of every ingestion run.  Accumulates across uploads so
    all previously indexed files remain searchable without re-ingesting them.
    video_clip chunks are skipped — their text is covered by the paired
    video_summary chunk.
    """
    global _index, _corpus_chunks

    new_slim = [
        _strip_binaries(c)
        for c in chunks
        if c.get("modality") not in _BM25_SKIP_MODALITIES
    ]
    _corpus_chunks.extend(new_slim)

    logger.info(
        f"Building BM25 index over {len(_corpus_chunks)} total chunks "
        f"(+{len(new_slim)} new)…"
    )

    if not _corpus_chunks:
        logger.warning("build_index: no indexable chunks after filtering — skipping BM25 build.")
        return

    tokenized_corpus = [_tokenize(c.get("text", "")) for c in _corpus_chunks]
    _index = BM25Okapi(tokenized_corpus)

    _BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_BM25_INDEX_PATH, "wb") as fh:
        pickle.dump((_PICKLE_VERSION, _index, _corpus_chunks), fh)

    logger.info(f"BM25 index persisted → {_BM25_INDEX_PATH} ({len(_corpus_chunks)} chunks)")


def load_index() -> None:
    """Load a previously persisted BM25 index from disk into memory.

    Called once at application startup by main.py.
    No-op if the index file does not exist yet.
    Discards the pickle and warns if the version stamp does not match
    _PICKLE_VERSION — this happens after a tokenizer change and requires
    re-ingesting files to rebuild.
    """
    global _index, _corpus_chunks
    if not _BM25_INDEX_PATH.exists():
        logger.warning("No BM25 index found on disk — will be created after first upload.")
        return
    with open(_BM25_INDEX_PATH, "rb") as fh:
        data = pickle.load(fh)

    if not isinstance(data, tuple) or data[0] != _PICKLE_VERSION:
        logger.warning(
            f"BM25 index version mismatch (expected {_PICKLE_VERSION}, "
            f"got {data[0] if isinstance(data, tuple) else 'unknown'}) — "
            "discarding stale index. Re-ingest files to rebuild."
        )
        return

    _, _index, _corpus_chunks = data
    logger.info(f"BM25 index loaded from disk ({len(_corpus_chunks)} chunks).")


def search_bm25(query: str, top_k: int = 20) -> list[dict]:
    """Return the top_k chunks most relevant to *query* via BM25 keyword scoring.

    Each result dict has keys: ``text``, ``score``, ``metadata``.
    Raises RuntimeError if the index has not been built yet.
    """
    if _index is None:
        raise RuntimeError("BM25 index is not loaded. Run an ingestion first.")

    logger.debug(f"BM25 search top_k={top_k}: {query!r}")
    tokenized_query = _tokenize(query)
    scores = _index.get_scores(tokenized_query)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results: list[dict] = []
    for idx in ranked:
        chunk = _corpus_chunks[idx]
        results.append(
            {
                "text": chunk.get("text", ""),
                "score": float(scores[idx]),
                "metadata": {
                    "source": chunk.get("source", ""),
                    "chunk_index": chunk.get("chunk_index", idx),
                    "page": chunk.get("page", 0),
                    "modality": chunk.get("modality", "text"),
                    "gcs_uri": chunk.get("gcs_uri", ""),
                },
            }
        )

    return results
