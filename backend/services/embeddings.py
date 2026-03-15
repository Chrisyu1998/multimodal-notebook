"""
Embeddings service — wraps the Gemini Embedding 2 API (google-genai SDK).

Key rules:
- Model: gemini-embedding-2-preview (natively multimodal)
- embed_text: single string, used only for query embedding at search time
- embed_chunks: handles image, video, audio, and document chunks
  - Images: batched up to 6 per request (hard API limit)
  - Documents: batched up to 250 per request (hard API limit), but the 20,000-token
    per-request cap is the binding constraint — default batch size is 20 (~800 tokens
    × 20 = 16,000 tokens, safely under the limit)
  - Video / audio: 1 per request, all concurrent
- Transient errors (429, 5xx, unknown) are retried with exponential backoff
- Permanent errors (4xx except 429) fail immediately — no wasted retries
- Failures raise EmbeddingBatchError with chunk_type + indices for checkpointing
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from google import genai
from google.genai import types
from loguru import logger

import backend.config as config

_client = genai.Client(api_key=config.GEMINI_API_KEY)

_IMAGE_BATCH_SIZE = config.EMBEDDING_IMAGE_BATCH_SIZE      # 6 — hard API limit
_DOCUMENT_BATCH_SIZE = config.EMBEDDING_DOCUMENT_BATCH_SIZE  # default 20 — bound by 20k token/request limit; max 250 inputs/request
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
_MAX_RETRIES = config.EMBEDDING_MAX_RETRIES
_MAX_WORKERS = config.EMBEDDING_MAX_WORKERS

# Types that must be sent one per API request
_SINGLE_FILE_TYPES = {"video", "audio"}

# Maps chunk type → bytes field name (document chunks embed text, not bytes)
_MEDIA_BYTES_FIELD: dict[str, str] = {
    "image": "image_bytes",
    "video": "video_bytes",
    "audio": "audio_bytes",
}

# Fixed MIME types for non-image media
_MEDIA_MIME: dict[str, str] = {
    "video": "video/mp4",
    "audio": "audio/mp3",
}

_RETRYABLE_STATUS_CODES = config.EMBEDDING_RETRYABLE_STATUS_CODES


class EmbeddingBatchError(Exception):
    """Raised when a batch fails after all retries.

    Carries enough context for the ingestion pipeline to checkpoint and
    re-queue just the affected chunks without parsing log messages.
    """

    def __init__(
        self,
        chunk_type: str,
        indices: list[int],
        cause: Exception,
        attempts: int = _MAX_RETRIES,
    ) -> None:
        self.chunk_type = chunk_type
        self.indices = indices
        self.cause = cause
        self.attempts = attempts
        super().__init__(
            f"Embedding failed for {chunk_type} chunks at indices {indices} "
            f"after {attempts} attempt(s): {cause}"
        )


def embed_text(text: str) -> list[float]:
    """Embed a single string — used for query embedding at search time."""
    return _with_retry(
        lambda: _client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=text,
        ).embeddings[0].values,
        chunk_type="query",
        indices=[-1],
    )


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add an 'embedding' key (list[float]) to each chunk dict in-place.

    Supports image, video, audio, and document chunks.
    Raises EmbeddingBatchError on failure — contains chunk_type and indices
    so the caller can checkpoint and re-queue exactly the affected chunks.
    """
    if not chunks:
        return chunks

    logger.info(f"Embedding {len(chunks)} chunks with {config.EMBEDDING_MODEL}…")

    image_items: list[tuple[int, dict]] = []
    single_items: list[tuple[int, dict]] = []
    text_items: list[tuple[int, dict]] = []

    for i, chunk in enumerate(chunks):
        chunk_type = chunk.get("type")
        if chunk_type == "image":
            image_items.append((i, chunk))
        elif chunk_type == "document":
            # Embed text directly. Synthetic PDFs (reportlab re-renders) and
            # video summary PDFs land in a different embedding subspace from
            # text queries — text→text alignment is exact.
            text_items.append((i, chunk))
        elif chunk_type in _SINGLE_FILE_TYPES:
            single_items.append((i, chunk))
        else:
            logger.warning(f"Chunk {i} has unknown type {chunk_type!r} — skipping")

    futures: dict = {}

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        # Images — up to 6 per call (hard API limit)
        for batch_start in range(0, len(image_items), _IMAGE_BATCH_SIZE):
            batch = image_items[batch_start : batch_start + _IMAGE_BATCH_SIZE]
            future = executor.submit(_embed_media_batch, batch)
            futures[future] = ("image", batch)

        # PDF text chunks — embed text field directly (batch up to 250 strings per call)
        for batch_start in range(0, len(text_items), _DOCUMENT_BATCH_SIZE):
            batch = text_items[batch_start : batch_start + _DOCUMENT_BATCH_SIZE]
            future = executor.submit(_embed_text_batch, batch)
            futures[future] = ("document", batch)

        # Video / audio — exactly 1 per call, all concurrent
        for i, chunk in single_items:
            future = executor.submit(_embed_media_batch, [(i, chunk)])
            futures[future] = (chunk["type"], [(i, chunk)])

        completed: dict[str, int] = {}

        for future in as_completed(futures):
            kind, batch = futures[future]
            indices = [i for i, _ in batch]
            try:
                embeddings = future.result()
            except EmbeddingBatchError:
                raise  # already has full context
            except Exception as exc:
                raise EmbeddingBatchError(kind, indices, exc) from exc

            for (i, chunk), embedding in zip(batch, embeddings):
                chunk["embedding"] = embedding
                chunks[i] = chunk

            completed[kind] = completed.get(kind, 0) + len(batch)
            logger.info(f"Embedded {completed[kind]} {kind} chunks…")

    return chunks


def _embed_media_batch(items: list[tuple[int, dict]]) -> list[list[float]]:
    """Embed a batch of media chunks in one API call.

    All items must share the same chunk type — callers are responsible for
    grouping by type before calling this function.
    Raises ValueError if mixed types are detected.
    """
    if not items:
        return []

    chunk_type = items[0][1]["type"]
    indices = [i for i, _ in items]

    for i, chunk in items:
        if chunk["type"] != chunk_type:
            raise ValueError(
                f"_embed_media_batch received mixed types: expected "
                f"{chunk_type!r}, got {chunk['type']!r} at index {i}"
            )

    def _call() -> list[list[float]]:
        parts = []
        for _, chunk in items:
            raw: bytes = chunk[_MEDIA_BYTES_FIELD[chunk_type]]
            mime_type = _image_mime(raw) if chunk_type == "image" else _MEDIA_MIME[chunk_type]
            parts.append(types.Part.from_bytes(data=raw, mime_type=mime_type))

        result = _client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=parts,
        )
        return [e.values for e in result.embeddings]

    return _with_retry(_call, chunk_type=chunk_type, indices=indices)


def _doc_embed_text(chunk: dict) -> str:
    """Return the string to embed for a document chunk.

    Prepends 'Doc: <title> | Section: <heading>' when those fields are present
    so the embedding captures document context without polluting chunk['text'],
    which is kept as clean raw text for generation and display.
    """
    title = chunk.get("document_title", "")
    section = chunk.get("section_heading", "")
    text = chunk.get("text", "")
    if title or section:
        return f"Doc: {title} | Section: {section}\n\n{text}"
    return text


def _embed_text_batch(items: list[tuple[int, dict]]) -> list[list[float]]:
    """Embed a batch of text-based chunks (PDF documents) using the text field.

    Sends all texts as a list of strings in a single API call.
    Stays within the 250-input / 20k-token-per-request limits when callers
    use _DOCUMENT_BATCH_SIZE (default 20).
    """
    if not items:
        return []

    indices = [i for i, _ in items]
    texts = [_doc_embed_text(chunk) for _, chunk in items]

    def _call() -> list[list[float]]:
        result = _client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=texts,
        )
        return [e.values for e in result.embeddings]

    return _with_retry(_call, chunk_type="document", indices=indices)


def _image_mime(data: bytes) -> str:
    """Return the MIME type for raw image bytes (jpeg or png)."""
    return "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"


def _extract_status_code(exc: Exception) -> Optional[int]:
    """Pull the HTTP status code from an exception, if present.

    OSError subclasses (ConnectionRefusedError, TimeoutError, etc.) are
    explicitly excluded — their .errno values are not HTTP status codes.
    gRPC status codes are mapped to their HTTP equivalents.
    All other exceptions without a recognisable status code return None,
    which _with_retry treats as transient (safe to retry).
    """
    if isinstance(exc, OSError):
        return None
    # google-genai SDK uses .code; other HTTP clients use .status_code
    if hasattr(exc, "code") and isinstance(exc.code, int):
        return exc.code
    if hasattr(exc, "status_code") and isinstance(exc.status_code, int):
        return exc.status_code
    grpc_to_http = {8: 429, 13: 500, 14: 503}
    if hasattr(exc, "grpc_status_code"):
        return grpc_to_http.get(exc.grpc_status_code.value)
    return None


def _with_retry(fn: Callable, chunk_type: str, indices: list[int]) -> list:
    """Run fn() with exponential backoff for transient errors.

    Permanent errors (4xx except 429) fail immediately — retrying bad data
    or auth failures wastes time and quota.
    Unknown status codes (including plain network errors) are treated as
    transient and retried.
    Raises EmbeddingBatchError if all attempts fail.
    """
    if _MAX_RETRIES < 1:
        raise EmbeddingBatchError(
            chunk_type, indices, ValueError(f"_MAX_RETRIES must be >= 1, got {_MAX_RETRIES}"), attempts=0
        )

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn()
        except EmbeddingBatchError:
            raise
        except Exception as exc:
            status_code = _extract_status_code(exc)

            if status_code is not None and status_code not in _RETRYABLE_STATUS_CODES:
                logger.error(
                    f"Permanent error (HTTP {status_code}) for {chunk_type} "
                    f"chunks {indices}: {exc}"
                )
                raise EmbeddingBatchError(chunk_type, indices, exc, attempts=attempt) from exc

            if attempt == _MAX_RETRIES:
                logger.error(
                    f"Exhausted {_MAX_RETRIES} retries for {chunk_type} "
                    f"chunks {indices}. Last error: {exc}"
                )
                raise EmbeddingBatchError(chunk_type, indices, exc, attempts=attempt) from exc

            wait = 2 ** (attempt - 1)
            logger.warning(
                f"{chunk_type} chunks {indices}: attempt {attempt}/{_MAX_RETRIES} "
                f"failed (HTTP {status_code or 'unknown'}): {exc}. "
                f"Retrying in {wait}s…"
            )
            time.sleep(wait)
