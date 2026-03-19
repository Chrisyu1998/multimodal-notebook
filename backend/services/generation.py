"""
Generation service — wraps Gemini 2.5 Flash for grounded multimodal answer generation.

Builds a strict RAG prompt: the model is constrained to answer only from
the provided context chunks and must say so when the answer is absent.

Modality handling at generation time:
  - PDF text / video_summary → plain text inline
  - image_global / image_local → caption text + JPEG/PNG inline_data from GCS
  - video_clip → SPEECH+VISUALS summary text + first-frame JPEG inline_data from GCS
GCS fetch failures fall back to text-only and are counted in media_chunks_degraded.
Gemini API errors are classified as retryable (429/503/network) or config errors
(401/400) so callers can map them to the correct HTTP status.
"""

import re

from google import genai
from google.genai import types
from loguru import logger

import backend.config as config
from backend.services import gcs

_client = genai.Client(api_key=config.GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GenerationError(RuntimeError):
    """Base class for all generation failures."""


class GenerationRetryableError(GenerationError):
    """Transient failure — rate limit (429), service unavailable (503), timeout.

    Callers should surface as HTTP 503 and may retry with backoff.
    """


class GenerationConfigError(GenerationError):
    """Permanent misconfiguration — invalid API key (401), bad request (400).

    Retrying will not help. Callers should surface as HTTP 500.
    """


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise research assistant answering questions from retrieved \
document sources. The sources may include text passages, images, and video keyframes.

INSTRUCTIONS:
- Answer using ONLY the provided sources. Do not use outside knowledge.
- Cite every claim inline with [N] where N is the source number.
- When the answer is not explicitly stated but can be derived by combining facts \
from multiple sources, synthesize them explicitly and cite each source used.
- If the sources do not contain enough information, say exactly: \
"I don't have enough information to answer this."
- Do not make up information.\
"""

# ---------------------------------------------------------------------------
# Modality routing constants
# ---------------------------------------------------------------------------

_IMAGE_MODALITIES: frozenset[str] = frozenset({"image", "image_global", "image_local"})
_VIDEO_CLIP_MODALITIES: frozenset[str] = frozenset({"video", "video_clip"})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RETRYABLE_PATTERNS: tuple[str, ...] = (
    "429",
    "quota",
    "rate limit",
    "503",
    "service unavailable",
    "deadline exceeded",
    "timeout",
)
_CONFIG_ERROR_PATTERNS: tuple[str, ...] = (
    "401",
    "403",
    "api key",
    "permission denied",
    "400",
    "invalid argument",
    "bad request",
)


def _classify_api_error(exc: Exception) -> GenerationError:
    """Map a raw Gemini SDK exception to a typed GenerationError subclass.

    Inspects the stringified exception for known status-code patterns.
    Defaults to GenerationRetryableError when the pattern is unrecognised
    so callers treat unknown failures as transient (safer than 500).
    """
    msg = str(exc).lower()
    for pattern in _CONFIG_ERROR_PATTERNS:
        if pattern in msg:
            return GenerationConfigError(str(exc))
    for pattern in _RETRYABLE_PATTERNS:
        if pattern in msg:
            return GenerationRetryableError(str(exc))
    # Unknown error — treat as retryable so callers can surface 503
    return GenerationRetryableError(str(exc))


def _build_context_parts(chunks: list[dict]) -> tuple[list[types.Part], int]:
    """Build an interleaved list of Parts from retrieved chunks.

    Each chunk contributes one or two Parts:
      - All modalities: one text Part (citation header + text content).
      - image_global / image_local: additionally one inline_data Part (JPEG/PNG
        bytes fetched from GCS) so the model reasons over actual pixels.
      - video_clip: additionally one inline_data Part (first-frame JPEG from GCS)
        for visual grounding. Only the representative frame is stored in GCS —
        not the full clip.
    GCS fetch failures are caught, logged, and counted. The affected chunk falls
    back to text-only so generation continues unblocked.

    Args:
        chunks: Top-k reranked chunk dicts (modality, gcs_uri, text, source, …).

    Returns:
        (parts, degraded_count) where parts is the flat list of types.Part
        objects and degraded_count is the number of media chunks that fell
        back to text due to GCS fetch failures.
    """
    parts: list[types.Part] = []
    degraded_count: int = 0

    for i, chunk in enumerate(chunks, start=1):
        modality: str = chunk.get("modality", "")
        gcs_uri: str = chunk.get("gcs_uri", "")
        source: str = chunk.get("source", "unknown")
        page: int = chunk.get("page", 0)
        text: str = chunk.get("text", "")

        # Build citation header — wording varies by modality for readability.
        if modality in _IMAGE_MODALITIES:
            citation = f"[{i}] Source: {source} (image)\nCaption: {text}"
        elif modality in _VIDEO_CLIP_MODALITIES and gcs_uri:
            citation = f"[{i}] Source: {source} (video clip)\n{text}"
        else:
            citation = f"[{i}] Source: {source}, Page: {page}\n{text}"

        # --- Image: caption text + inline image bytes ---
        if modality in _IMAGE_MODALITIES and gcs_uri:
            try:
                img_bytes = gcs.download_bytes(gcs_uri)
                mime = "image/png" if gcs_uri.endswith(".png") else "image/jpeg"
                parts.append(types.Part.from_text(text=citation))
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
                logger.debug(
                    f"generate: chunk {i} — image inline_data "
                    f"({len(img_bytes):,} bytes, {mime})"
                )
                continue
            except Exception as exc:
                logger.warning(
                    f"generate: GCS fetch failed for image chunk {i} "
                    f"({gcs_uri}): {exc} — falling back to text"
                )
                degraded_count += 1
                parts.append(
                    types.Part.from_text(text=f"[{i}] Source: {source} (image)\nCaption: {text}")
                )
                continue

        # --- Video clip: summary text + first-frame JPEG ---
        elif modality in _VIDEO_CLIP_MODALITIES and gcs_uri:
            try:
                frame_bytes = gcs.download_bytes(gcs_uri)
                parts.append(types.Part.from_text(text=citation))
                parts.append(
                    types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg")
                )
                logger.debug(
                    f"generate: chunk {i} — video frame inline_data "
                    f"({len(frame_bytes):,} bytes)"
                )
                continue
            except Exception as exc:
                logger.warning(
                    f"generate: GCS fetch failed for video frame chunk {i} "
                    f"({gcs_uri}): {exc} — falling back to text"
                )
                degraded_count += 1
                parts.append(
                    types.Part.from_text(text=f"[{i}] Source: {source} (video clip)\n{text}")
                )
                continue

        # --- Text-only path: PDF, video_summary, or GCS fallback ---
        parts.append(types.Part.from_text(text=citation))

    return parts, degraded_count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Call Gemini 2.5 Flash with a grounded multimodal RAG prompt.

    Constructs a multipart Content list combining text citations with inline
    media (images and video frames fetched from GCS). PDF chunks
    pass transcript/text only. Native thinking (thinking_config) is used for
    chain-of-thought reasoning instead of prompted <thinking> blocks, keeping
    the response clean and avoiding format-parsing fragility.

    Args:
        question: The user's natural language question.
        chunks: Top-k reranked chunk dicts from retrieval, each with keys:
                text, source, page, modality, gcs_uri, score.

    Returns:
        Dict with keys: answer (str), sources (list[dict]), chunks_used (int),
        model (str), media_chunks_degraded (int).

    Raises:
        GenerationRetryableError: Transient API failure (rate limit, 503, timeout).
        GenerationConfigError: Permanent misconfiguration (bad API key, 400).
    """
    for i, chunk in enumerate(chunks, start=1):
        preview = (chunk.get("text") or "")[:120].replace("\n", " ")
        logger.info(
            f"  Chunk {i} [{chunk.get('source')} p{chunk.get('page')} "
            f"modality={chunk.get('modality', '?')}]: {preview!r}"
        )
    logger.debug(
        f"Calling {config.GENERATION_MODEL} with {len(chunks)} context chunks "
        f"(thinking_budget={config.GENERATION_THINKING_BUDGET})"
    )

    context_parts, degraded_count = _build_context_parts(chunks)
    if degraded_count:
        logger.warning(
            f"generate: {degraded_count}/{len(chunks)} media chunk(s) degraded to text"
        )

    contents: list[types.Part] = (
        [types.Part.from_text(text="Context:\n")]
        + context_parts
        + [types.Part.from_text(text=f"\n\nQuestion: {question}\n\nAnswer:")]
    )

    try:
        response = _client.models.generate_content(
            model=config.GENERATION_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=1.0,  # required when thinking is enabled
                thinking_config=types.ThinkingConfig(
                    thinking_budget=config.GENERATION_THINKING_BUDGET,
                ),
            ),
        )
    except Exception as exc:
        logger.error(f"Gemini generation API failed: {exc}")
        raise _classify_api_error(exc) from exc

    # Log native thinking output for debugging (thought parts are separate from
    # the answer in the Gemini response; response.text already excludes them).
    for part in response.candidates[0].content.parts:
        if getattr(part, "thought", False):
            logger.debug(f"generate: thinking block:\n{part.text}")

    answer_text = response.text.strip()

    # Token usage — available on response.usage_metadata (may be None for some
    # model versions; guard with getattr so callers never see a missing attribute).
    usage = getattr(response, "usage_metadata", None)
    input_tokens: int = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens: int = int(getattr(usage, "candidates_token_count", 0) or 0)
    logger.debug(
        f"generate: token usage — input={input_tokens} output={output_tokens}"
    )

    # Citation validation — warn if the model references a [N] that doesn't
    # correspond to any retrieved chunk (hallucinated source reference).
    cited_indices = {int(m) for m in re.findall(r"\[(\d+)\]", answer_text)}
    valid_indices = set(range(1, len(chunks) + 1))
    hallucinated = cited_indices - valid_indices
    if hallucinated:
        logger.warning(
            f"generate: hallucinated source reference(s) detected: "
            f"{sorted(hallucinated)} (only {len(chunks)} chunk(s) available)"
        )

    sources = [
        {
            "filename": chunk.get("source", "unknown"),
            "page": chunk.get("page", 0),
            "score": chunk.get("score", 0.0),
            "snippet": (chunk.get("text") or "")[:100],
        }
        for chunk in chunks
    ]

    return {
        "answer": answer_text,
        "sources": sources,
        "chunks_used": len(chunks),
        "model": config.GENERATION_MODEL,
        "media_chunks_degraded": degraded_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
