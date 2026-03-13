"""
Embeddings service — wraps the Gemini Embedding 2 API.

Key rules:
- Always batch in groups of 100 — never call the API one chunk at a time
- Model: config.EMBEDDING_MODEL  ("text-embedding-004")
- Text chunks   → embed the "text" field
- Image chunks  → pass raw bytes directly (multimodal embedding)
- Video frames  → same as image chunks
"""

from loguru import logger

import backend.config as config

# TODO: import google.generativeai and configure with config.GEMINI_API_KEY


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an "embedding" key (list[float]) to each chunk dict in-place.
    Dispatches to _embed_text_batch or _embed_image for each chunk type.
    Batches text chunks in groups of 100 to stay within API rate limits.
    Returns the same list with embeddings populated.
    """
    logger.info(f"Embedding {len(chunks)} chunks…")
    # TODO: separate chunks into text vs image/video_frame groups
    # TODO: call _embed_text_batch for text chunks in groups of 100
    # TODO: call _embed_image for each image/video_frame chunk
    # TODO: attach embedding back onto each chunk dict
    raise NotImplementedError


def embed_text(text: str) -> list[float]:
    """
    Embed a single string — used for query embedding (HyDE).
    Returns a flat list of floats.
    """
    logger.debug("embed_text called")
    # TODO: call Gemini Embedding 2 with model=config.EMBEDDING_MODEL
    # TODO: return embedding vector as list[float]
    raise NotImplementedError


def _embed_text_batch(texts: list[str]) -> list[list[float]]:
    """
    Internal helper: embed a list of strings in one API call.
    Caller is responsible for capping batch size at 100.
    """
    # TODO: call genai.embed_content(model=..., content=texts, ...)
    # TODO: return list of embedding vectors
    raise NotImplementedError


def _embed_image(image_bytes: bytes) -> list[float]:
    """
    Internal helper: embed a single image using the multimodal endpoint.
    """
    # TODO: call genai.embed_content with image Part
    # TODO: return embedding vector as list[float]
    raise NotImplementedError
