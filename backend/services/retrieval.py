"""
Retrieval service — HyDE query expansion, Reciprocal Rank Fusion, and Gemini reranking.

Pipeline order (called from query.py):
  1. hyde_expand   — generate a hypothetical answer and embed it instead of the raw
                     query; produces a richer query vector that sits closer to answer
                     space than question space.
  2. hybrid_search — call BM25 (keyword) and vector (semantic) search in parallel,
                     then fuse results with Reciprocal Rank Fusion.
  3. rerank        — ask Gemini Flash to score the top-20 fused candidates and keep
                     only the top_k most relevant.

Design notes:
  - RRF is implemented from scratch (no LangChain).  Every line is annotated so
    the logic is interview-explainable.
  - Deduplication key is exact chunk text.  Two chunks with identical text from
    different result lists are treated as the same document and their rank
    contributions are summed before sorting.
  - BM25 results carry a nested 'metadata' dict; vector results are flat.
    _normalize() collapses both into the same shape so RRF can treat them uniformly.
"""

import json
from typing import Any

from google import genai
from google.genai import types
from loguru import logger

import backend.config as config
from backend.services import bm25_index, embeddings, vectorstore

_client = genai.Client(api_key=config.GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# RRF constant
# ---------------------------------------------------------------------------

# k=60 is the standard constant from the original RRF paper (Cormack et al., 2009).
# It dampens the impact of very high-ranked documents so that a #1 result in one
# list doesn't completely dominate over a consistent #3/#4 across both lists.
_RRF_K: int = 60


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(result: dict) -> dict:
    """Flatten a result dict into a canonical shape consumed by RRF.

    BM25 results arrive as:
        {"text": ..., "score": ..., "metadata": {"source": ..., "chunk_index": ...,
                                                  "page": ..., "modality": ...}}

    Vector results arrive as:
        {"text": ..., "source": ..., "page": ..., "chunk_index": ...,
         "type": ..., "score": ...}

    Output (canonical):
        {"text": ..., "source": ..., "page": ..., "chunk_index": ...,
         "modality": ..., "orig_score": ...}

    'orig_score' preserves the raw BM25/cosine score for logging; the caller
    replaces 'score' with the RRF score before returning to callers upstream.
    """
    if "metadata" in result:
        # BM25 shape
        meta: dict = result["metadata"]
        return {
            "text": result.get("text", ""),
            "source": meta.get("source", ""),
            "page": meta.get("page", 0),
            "chunk_index": meta.get("chunk_index", 0),
            "modality": meta.get("modality", "text"),
            "orig_score": result.get("score", 0.0),
        }
    else:
        # Vector shape
        return {
            "text": result.get("text", ""),
            "source": result.get("source", ""),
            "page": result.get("page", 0),
            "chunk_index": result.get("chunk_index", 0),
            "modality": result.get("type", result.get("modality", "")),
            "orig_score": result.get("score", 0.0),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    top_k: int = 20,
) -> list[dict]:
    """Merge two ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF formula (Cormack et al., 2009):
        rrf_score(doc) = Σ  1 / (k + rank_i)
                         i
    where k=60 is a smoothing constant and rank_i is the 1-indexed position of
    the document in result list i.  Documents that appear in only one list still
    receive a contribution from that list; documents present in both lists get
    the sum of both contributions, which pushes consistently-ranked content up.

    Deduplication is by exact text string.  If two chunks from different lists
    share the same text, they are merged into one entry and their rank
    contributions are accumulated.  Metadata from the first-seen occurrence is
    kept (BM25 and vector should agree on source/page for the same chunk).

    Args:
        bm25_results:   Ranked list from search_bm25(), highest rank first.
        vector_results: Ranked list from vectorstore.search(), highest rank first.
        top_k:          Maximum number of results to return.

    Returns:
        Deduplicated list sorted by descending RRF score, capped at top_k.
        Each dict has keys: text, source, page, chunk_index, modality, score
        (the RRF score), orig_bm25_rank, orig_vector_rank (for debugging).
    """
    # ---- Step 1: Normalise both lists to a common dict shape ----
    # This decouples RRF logic from the quirky shapes of the upstream services.
    bm25_norm   = [_normalize(r) for r in bm25_results]
    vector_norm = [_normalize(r) for r in vector_results]

    # ---- Step 2: Build a registry keyed by exact chunk text ----
    # registry maps  text → {"meta": canonical_dict, "rrf_score": float,
    #                         "bm25_rank": int|None, "vector_rank": int|None}
    # We use text as the dedup key because it's the only field guaranteed to
    # match across the two retrieval backends (IDs are not shared).
    registry: dict[str, dict[str, Any]] = {}

    # ---- Step 3: Accumulate BM25 rank contributions ----
    for rank_0indexed, doc in enumerate(bm25_norm):
        rank = rank_0indexed + 1                        # RRF uses 1-indexed ranks
        contribution = 1.0 / (_RRF_K + rank)           # core RRF formula

        text = doc["text"]
        if text not in registry:
            registry[text] = {
                "meta": doc,
                "rrf_score": 0.0,
                "bm25_rank": None,
                "vector_rank": None,
            }

        registry[text]["rrf_score"] += contribution    # add this list's contribution
        registry[text]["bm25_rank"] = rank             # record rank for debugging

    # ---- Step 4: Accumulate vector rank contributions ----
    # Identical logic to Step 3 — if the text already exists in the registry
    # (i.e. it appeared in the BM25 list too), the contribution is *added* to
    # whatever BM25 already contributed.  That's the key insight of RRF: docs
    # that rank well in multiple lists get a boost, not a replacement.
    for rank_0indexed, doc in enumerate(vector_norm):
        rank = rank_0indexed + 1
        contribution = 1.0 / (_RRF_K + rank)

        text = doc["text"]
        if text not in registry:
            registry[text] = {
                "meta": doc,
                "rrf_score": 0.0,
                "bm25_rank": None,
                "vector_rank": None,
            }

        registry[text]["rrf_score"] += contribution
        registry[text]["vector_rank"] = rank

    # ---- Step 5: Sort by descending RRF score and trim to top_k ----
    sorted_entries = sorted(
        registry.values(),
        key=lambda entry: entry["rrf_score"],
        reverse=True,
    )[:top_k]

    # ---- Step 6: Build output dicts — flatten meta + attach rrf_score ----
    results: list[dict] = []
    for entry in sorted_entries:
        meta = entry["meta"]
        results.append({
            "text":          meta["text"],
            "source":        meta["source"],
            "page":          meta["page"],
            "chunk_index":   meta["chunk_index"],
            "modality":      meta["modality"],
            "score":         entry["rrf_score"],       # downstream uses 'score' uniformly
            # Debug fields — visible in logs but not returned to the UI
            "_bm25_rank":    entry["bm25_rank"],
            "_vector_rank":  entry["vector_rank"],
        })

    return results


def hyde_expand(query: str) -> str:
    """Generate a hypothetical document passage for HyDE (Hypothetical Document Embeddings).

    Instead of embedding the raw question ("What is X?"), we ask the LLM to write
    a 2-3 sentence passage as if it existed in the corpus.  The resulting text sits
    in *answer space* rather than *question space*, dramatically improving cosine
    similarity against real answer chunks.

    The prompt instructs the model to match the style and vocabulary of a real source
    document so the hypothetical embedding lands close to genuine corpus embeddings.

    Falls back to the original query on any API error so the pipeline never blocks.
    """
    prompt = (
        "Write a short passage (2-3 sentences) that directly answers the question "
        "below. Write it in the style and vocabulary of a document excerpt — as if "
        "it were a paragraph taken from a real source document, textbook, or report. "
        "Do not say 'I' or acknowledge that you are generating text; just write the "
        "passage.\n\n"
        f"Question: {query}"
    )

    try:
        response = _client.models.generate_content(
            model=config.GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        hypothetical = response.text.strip()
        logger.info(f"HyDE hypothetical doc | query={query!r} | doc={hypothetical!r}")
        return hypothetical
    except Exception as exc:
        logger.warning(f"HyDE expansion failed ({exc}) — falling back to raw query.")
        return query


def hybrid_search(query: str, top_k: int = 20, use_hyde: bool = True) -> list[dict]:
    """Run BM25 + vector search and fuse results with Reciprocal Rank Fusion.

    Steps:
      1. (Optional) HyDE: generate a hypothetical document and embed it for vector
         search.  When use_hyde=False the raw query is embedded directly instead.
         BM25 always uses the *original* query because keyword matching works
         best on the literal terms the user typed.
      2. BM25 keyword search  → top config.BM25_TOP_K results
      3. Vector semantic search → top config.VECTOR_TOP_K results
      4. RRF fusion            → merged, deduplicated list sorted by RRF score
      5. Return top_k results

    Args:
        query:     The user's original natural-language question.
        top_k:     Number of results to return (default 20, trimmed after RRF).
        use_hyde:  If True (default), expand the query via HyDE before embedding.
                   Set to False to embed the raw query — useful for A/B comparison.

    Returns:
        Merged result list with keys: text, source, page, chunk_index,
        modality, score (RRF score).
    """
    logger.info(f"hybrid_search top_k={top_k} use_hyde={use_hyde}: {query!r}")

    # ---- HyDE: embed a hypothetical document rather than the bare question ----
    # When use_hyde=False we skip the LLM call and embed the raw query directly.
    # BM25 always receives the original query regardless of this flag.
    if use_hyde:
        hyde_text = hyde_expand(query)
    else:
        hyde_text = query
        logger.debug("HyDE disabled — embedding raw query for vector search.")
    query_embedding = embeddings.embed_text(hyde_text)

    # ---- BM25: keyword search on the original query ----
    # BM25 uses the literal question because Porter-stemmed token overlap is
    # what it optimises for; a hypothetical *answer* adds noise here.
    try:
        bm25_results = bm25_index.search_bm25(query, top_k=config.BM25_TOP_K)
        logger.debug(f"BM25 returned {len(bm25_results)} results")
        for i, r in enumerate(bm25_results):
            meta = r.get("metadata", {})
            logger.debug(
                f"  BM25[{i+1}] score={r.get('score', 0):.4f} "
                f"source={meta.get('source', '?')} page={meta.get('page', '?')} "
                f"| {r.get('text', '')[:120]!r}"
            )
    except RuntimeError as exc:
        # Index not built yet (no uploads) — degrade gracefully to vector-only
        logger.warning(f"BM25 unavailable ({exc}) — using vector search only.")
        bm25_results = []

    # ---- Vector: semantic search on the HyDE embedding ----
    vector_results = vectorstore.search(query_embedding, top_k=config.VECTOR_TOP_K)
    logger.debug(f"Vector search returned {len(vector_results)} results")
    for i, r in enumerate(vector_results):
        logger.debug(
            f"  Vector[{i+1}] score={r.get('score', 0):.4f} "
            f"source={r.get('source', '?')} page={r.get('page', '?')} "
            f"| {r.get('text', '')[:120]!r}"
        )

    # ---- RRF: merge both ranked lists ----
    merged = reciprocal_rank_fusion(bm25_results, vector_results, top_k=top_k)
    logger.info(
        f"RRF merged {len(bm25_results)} BM25 + {len(vector_results)} vector "
        f"→ {len(merged)} unique chunks (top_k={top_k})"
    )
    for i, r in enumerate(merged):
        logger.debug(
            f"  RRF[{i + 1}] score={r.get('score', 0):.4f} "
            f"bm25_rank={r.get('_bm25_rank', '-')} vector_rank={r.get('_vector_rank', '-')} "
            f"source={r.get('source', '?')} page={r.get('page', '?')} "
            f"| {r.get('text', '')[:120]!r}"
        )

    return merged


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Score all candidate chunks in a single Gemini Flash call, keep top_k.

    Sends one prompt containing all chunks so the model scores them relative to
    each other — consistent calibration across the full candidate set.  Falls
    back to the existing RRF order if the API call or JSON parsing fails.

    Prompt asks for ONLY a JSON array:
        [{"id": 1, "score": 0.95}, {"id": 2, "score": 0.42}, ...]
    where id is 1-indexed and score is 0.0–1.0 (1.0 = perfectly relevant).

    Args:
        query:   The original user question.
        chunks:  Candidate chunks from hybrid_search(), up to 20.
        top_k:   How many chunks to keep (default 5, fed to generation).

    Returns:
        Up to top_k chunk dicts sorted by rerank_score descending.
        Each dict gains a 'rerank_score' field (float, 0.0–1.0).
    """
    if not chunks:
        return []
    if len(chunks) <= top_k:
        for c in chunks[:top_k]:
            c["rerank_score"] = c.get("score", 0.0)
        return chunks[:top_k]

    # Each passage is truncated to 400 chars — enough signal for relevance
    # scoring while keeping the prompt well within Flash's context window.
    passages = "\n\n".join(
        f"[{i + 1}] {c['text'][:400]}" for i, c in enumerate(chunks)
    )
    prompt = (
        "You are a relevance reranker for a retrieval-augmented generation system.\n\n"
        "Score each passage below for how relevant it is to answering the query.\n\n"
        "Respond with ONLY a JSON array — no explanation, no markdown fences:\n"
        '[{"id": <1-indexed int>, "score": <float 0.0–1.0>}, ...]\n\n'
        "Where 1.0 = directly and completely answers the query,\n"
        "      0.5 = partially relevant,\n"
        "      0.0 = completely unrelated.\n\n"
        f"Query: {query}\n\n"
        f"Passages:\n{passages}"
    )

    try:
        response = _client.models.generate_content(
            model=config.RERANK_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        raw = response.text.strip()
        # Strip markdown fences if the model wraps the array in ```json … ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        scores: list[dict] = json.loads(raw)
        id_to_score: dict[int, float] = {
            int(item["id"]): float(item["score"]) for item in scores
        }
    except Exception as exc:
        logger.warning(f"Reranker failed ({exc}) — falling back to RRF order.")
        for c in chunks[:top_k]:
            c["rerank_score"] = c.get("score", 0.0)
        return chunks[:top_k]

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = id_to_score.get(i + 1, 0.0)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    logger.info(
        f"rerank: top {top_k}/{len(chunks)} scores: "
        f"{[round(c['rerank_score'], 3) for c in ranked[:top_k]]}"
    )
    for i, c in enumerate(ranked[:top_k]):
        logger.debug(
            f"  [{i + 1}] rerank={c['rerank_score']:.3f} rrf={c.get('score', 0):.4f} "
            f"src={c.get('source', '?')} pg={c.get('page', '?')} "
            f"| {c['text'][:100]!r}"
        )
    return ranked[:top_k]
