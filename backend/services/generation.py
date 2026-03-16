"""
Generation service — wraps Gemini 1.5 Pro for grounded answer generation.

Builds a strict RAG prompt: the model is constrained to answer only from
the provided context chunks and must say so when the answer is absent.
"""

from google import genai
from google.genai import types
from loguru import logger

import backend.config as config

_client = genai.Client(api_key=config.GEMINI_API_KEY)


class GenerationError(RuntimeError):
    """Raised when the Gemini generation API call fails (auth, rate limit, etc.)."""


_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the context "
    "provided below. If the answer is not in the context, say "
    '"I don\'t have enough information to answer this." '
    "Do not make up information."
)


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Call Gemini 1.5 Pro with a grounded RAG prompt and return the answer dict.

    Args:
        question: The user's natural language question.
        chunks: Top-k retrieved chunk dicts, each with keys: text, source, page, score.

    Returns:
        Dict with keys: answer (str), sources (list[dict]), chunks_used (int), model (str).
    """
    context_lines = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", 0)
        text = chunk.get("text", "")
        context_lines.append(f"[{i}] Source: {source}, Page: {page}\n{text}")

    context_block = "\n\n".join(context_lines)

    user_message = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    for i, chunk in enumerate(chunks, start=1):
        preview = (chunk.get("text") or "")[:120].replace("\n", " ")
        logger.info(f"  Chunk {i} [{chunk.get('source')} p{chunk.get('page')}]: {preview!r}")
    logger.debug(f"Calling {config.GENERATION_MODEL} with {len(chunks)} context chunks")

    try:
        response = _client.models.generate_content(
            model=config.GENERATION_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )
    except Exception as exc:
        logger.error(f"Gemini generation API failed: {exc}")
        raise GenerationError(str(exc)) from exc

    answer_text = response.text.strip()

    sources = [
        {
            "filename": chunk.get("source", "unknown"),
            "page": chunk.get("page", 0),
            "score": chunk.get("score", 0.0),
        }
        for chunk in chunks
    ]

    return {
        "answer": answer_text,
        "sources": sources,
        "chunks_used": len(chunks),
        "model": config.GENERATION_MODEL,
    }
