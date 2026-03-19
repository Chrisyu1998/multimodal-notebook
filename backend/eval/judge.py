"""
LLM-as-judge scoring functions for RAG eval.

Each function calls config.EVAL_JUDGE_MODEL (gemini-2.5-pro) with a structured
prompt and returns {"score": float, "reasoning": str}.  All API calls are
synchronous and intended to be offloaded via asyncio.to_thread by the runner.
"""

import json
import time

from google import genai
from google.genai import types
from loguru import logger

import backend.config as config

# ---------------------------------------------------------------------------
# Shared client
# ---------------------------------------------------------------------------

_client = genai.Client(api_key=config.GEMINI_API_KEY)
_JUDGE_MAX_RETRIES: int = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_chunks(chunks: list[dict]) -> str:
    """Return numbered text representation of retrieved chunks for prompts."""
    if not chunks:
        return "(no chunks retrieved)"
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", 0)
        text = chunk.get("text", "").strip()
        lines.append(f"[{i}] {source} p{page}: {text}")
    return "\n\n".join(lines)


def _call_judge(prompt: str) -> str:
    """Call EVAL_JUDGE_MODEL with exponential backoff retry on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(_JUDGE_MAX_RETRIES):
        try:
            response = _client.models.generate_content(
                model=config.EVAL_JUDGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            text = response.text.strip()
            # Strip markdown code fences that thinking models often emit
            if text.startswith("```"):
                lines = text.splitlines()
                # Drop opening fence line (```json or ```)
                lines = lines[1:]
                # Drop closing fence line
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            return text
        except Exception as exc:
            last_exc = exc
            if attempt < _JUDGE_MAX_RETRIES - 1:
                wait = 2 ** attempt  # 1s, then 2s
                logger.warning(
                    f"judge API call failed (attempt {attempt + 1}/{_JUDGE_MAX_RETRIES}): "
                    f"{exc} — retrying in {wait}s"
                )
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_correctness(
    query: str,
    ground_truth: str,
    generated_answer: str,
    retrieved_chunks: list[dict],  # uniform interface; correctness grades against ground_truth only
) -> dict:
    """Score factual accuracy of generated_answer vs ground_truth on a 0-5 scale."""
    prompt = f"""\
You are an expert evaluator assessing the factual accuracy of a generated answer.

Query:
{query}

Ground truth answer:
{ground_truth}

Generated answer:
{generated_answer}

Reason step by step through each factual claim in the generated answer, comparing \
it to the ground truth. Then assign a score from 0 to 5. Fractional scores such as \
3.5 or 4.5 are encouraged when the answer falls between two anchor points:
  5 — Perfectly correct; all facts match ground truth.
  4 — Mostly correct; minor imprecision or one small omission.
  3 — Partially correct; one clearly wrong or missing key detail.
  2 — Partially correct but significant factual errors.
  1 — Mostly wrong; only minor elements are correct.
  0 — Completely wrong or directly contradicts ground truth.

Partial credit is important: do not assign 0 unless the answer is entirely wrong.

Return ONLY valid JSON with no preamble and no markdown fences:
{{"score": <float 0.0-5.0>, "reasoning": "<step-by-step reasoning>"}}"""

    raw = _call_judge(prompt)
    try:
        data = json.loads(raw)
        score = max(0.0, min(5.0, float(data["score"])))
        reasoning = str(data["reasoning"])
    except (json.JSONDecodeError, KeyError, ValueError):
        score = 0.0
        reasoning = f"parse_error: {raw[:200]}"

    logger.debug(f"correctness | query={query[:80]!r} | score={score}")
    return {"score": score, "reasoning": reasoning}


def score_hallucination(
    query: str,
    ground_truth: str,  # uniform interface; hallucination checks against retrieved_chunks only
    generated_answer: str,
    retrieved_chunks: list[dict],
) -> dict:
    """Return hallucination rate (0.0 = fully grounded, 1.0 = fully hallucinated)."""
    chunk_text = _format_chunks(retrieved_chunks)
    prompt = f"""\
You are an expert evaluator checking whether a generated answer is grounded \
in retrieved context.

Query:
{query}

Retrieved context (the ONLY permitted source — do NOT use your own knowledge):
{chunk_text}

Generated answer:
{generated_answer}

Identify every factual claim in the generated answer. For each claim decide \
whether it is directly supported by the retrieved context above.
Then compute:
  hallucination_rate = unsupported_claims / total_claims
  0.0 means every claim is supported (no hallucination).
  1.0 means no claim is supported (complete hallucination).
  If there are no claims, return 0.0.

Return ONLY valid JSON with no preamble and no markdown fences:
{{"hallucination_rate": <float 0.0-1.0>, "reasoning": "<per-claim analysis>"}}"""

    raw = _call_judge(prompt)
    try:
        data = json.loads(raw)
        score = max(0.0, min(1.0, float(data["hallucination_rate"])))
        reasoning = str(data["reasoning"])
    except (json.JSONDecodeError, KeyError, ValueError):
        score = 0.0
        reasoning = f"parse_error: {raw[:200]}"

    logger.debug(f"hallucination | query={query[:80]!r} | rate={score}")
    return {"score": score, "reasoning": reasoning}


def score_faithfulness(
    query: str,
    ground_truth: str,  # uniform interface; faithfulness checks chunk usage, not ground_truth
    generated_answer: str,
    retrieved_chunks: list[dict],
) -> dict:
    """Score how faithfully the answer is derived from retrieved chunks (0-5)."""
    chunk_text = _format_chunks(retrieved_chunks)
    prompt = f"""\
You are an expert evaluator assessing whether a generated answer is derived \
from retrieved context rather than the model's general knowledge.

Query:
{query}

Retrieved chunks:
{chunk_text}

Generated answer:
{generated_answer}

Score faithfulness on a 0-5 scale:
  5 — Answer is clearly and directly derived from the retrieved chunks.
  4 — Mostly derived from chunks; minor use of general knowledge.
  3 — Roughly half derived from chunks, half from general knowledge.
  2 — Mostly uses general knowledge; retrieved chunks barely referenced.
  1 — Retrieved chunks almost entirely ignored.
  0 — Answer ignores chunks entirely and relies only on general knowledge.

Return ONLY valid JSON with no preamble and no markdown fences:
{{"score": <float 0-5>, "reasoning": "<analysis>"}}"""

    raw = _call_judge(prompt)
    try:
        data = json.loads(raw)
        score = max(0.0, min(5.0, float(data["score"])))
        reasoning = str(data["reasoning"])
    except (json.JSONDecodeError, KeyError, ValueError):
        score = 0.0
        reasoning = f"parse_error: {raw[:200]}"

    logger.debug(f"faithfulness | query={query[:80]!r} | score={score}")
    return {"score": score, "reasoning": reasoning}


def score_context_precision(
    query: str,
    ground_truth: str,
    generated_answer: str,
    retrieved_chunks: list[dict],
) -> dict:
    """Return fraction of retrieved chunks that were relevant to the query (0.0-1.0)."""
    if not retrieved_chunks:
        logger.debug(f"context_precision | query={query[:80]!r} | precision=0.0 (no chunks)")
        return {"score": 0.0, "reasoning": "No chunks retrieved."}

    chunk_text = _format_chunks(retrieved_chunks)
    total = len(retrieved_chunks)
    prompt = f"""\
You are an expert evaluator assessing retrieval precision.

Query:
{query}

Retrieved chunks (evaluate each independently):
{chunk_text}

For each of the {total} chunk(s) above, decide: is this chunk relevant to \
answering the query? A chunk is relevant if it contains information that \
directly helps answer the question.

Compute: precision = relevant_count / {total}

Return ONLY valid JSON with no preamble and no markdown fences:
{{"precision": <float 0.0-1.0>, "reasoning": "<per-chunk relevance assessment>"}}"""

    raw = _call_judge(prompt)
    try:
        data = json.loads(raw)
        score = max(0.0, min(1.0, float(data["precision"])))
        reasoning = str(data["reasoning"])
    except (json.JSONDecodeError, KeyError, ValueError):
        score = 0.0
        reasoning = f"parse_error: {raw[:200]}"

    logger.debug(f"context_precision | query={query[:80]!r} | precision={score}")
    return {"score": score, "reasoning": reasoning}
