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
  - Reranker sends a single multimodal Gemini request so all candidates are scored
    relative to each other.  Image/video chunks pass their actual media bytes
    (or a pre-extracted JPEG frame for video) instead of caption text, eliminating
    the cross-modal embedding bias that would otherwise penalise non-text chunks.
"""

import json
import re
from typing import Any

from google import genai
from google.genai import types
from loguru import logger

import backend.config as config
from backend.services import bm25_index, embeddings, gcs, vectorstore

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
                                                  "page": ..., "modality": ...,
                                                  "gcs_uri": ...}}

    Vector results arrive as:
        {"text": ..., "source": ..., "page": ..., "chunk_index": ...,
         "type": ..., "score": ..., "gcs_uri": ...}

    Output (canonical):
        {"text": ..., "source": ..., "page": ..., "chunk_index": ...,
         "modality": ..., "orig_score": ..., "gcs_uri": ...}

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
            "gcs_uri": meta.get("gcs_uri", ""),
        }
    else:
        # Vector shape — prefer specific modality (e.g. "image_global") over
        # the broad type field (e.g. "image") so generation can route correctly.
        return {
            "text": result.get("text", ""),
            "source": result.get("source", ""),
            "page": result.get("page", 0),
            "chunk_index": result.get("chunk_index", 0),
            "modality": result.get("modality", "") or result.get("type", ""),
            "orig_score": result.get("score", 0.0),
            "gcs_uri": result.get("gcs_uri", ""),
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
        (the RRF score), gcs_uri, orig_bm25_rank, orig_vector_rank (for debugging).
    """
    # ---- Step 1: Normalise both lists to a common dict shape ----
    bm25_norm   = [_normalize(r) for r in bm25_results]
    vector_norm = [_normalize(r) for r in vector_results]

    # ---- Step 2: Build a registry keyed by exact chunk text ----
    # registry maps  text → {"meta": canonical_dict, "rrf_score": float,
    #                         "bm25_rank": int|None, "vector_rank": int|None}
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

        registry[text]["rrf_score"] += contribution
        registry[text]["bm25_rank"] = rank

    # ---- Step 4: Accumulate vector rank contributions ----
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
            "score":         entry["rrf_score"],
            "gcs_uri":       meta.get("gcs_uri", ""),
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

    Returns:
        Merged result list with keys: text, source, page, chunk_index,
        modality, score (RRF score), gcs_uri.
    """
    logger.info(f"hybrid_search top_k={top_k} use_hyde={use_hyde}: {query!r}")

    if use_hyde:
        hyde_text = hyde_expand(query)
    else:
        hyde_text = query
        logger.debug("HyDE disabled — embedding raw query for vector search.")
    query_embedding = embeddings.embed_text(hyde_text)

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
        logger.warning(f"BM25 unavailable ({exc}) — using vector search only.")
        bm25_results = []

    vector_results = vectorstore.search(query_embedding, top_k=config.VECTOR_TOP_K)
    logger.debug(f"Vector search returned {len(vector_results)} results")
    for i, r in enumerate(vector_results):
        logger.debug(
            f"  Vector[{i+1}] score={r.get('score', 0):.4f} "
            f"source={r.get('source', '?')} page={r.get('page', '?')} "
            f"| {r.get('text', '')[:120]!r}"
        )

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


# ---------------------------------------------------------------------------
# Modality routing
# ---------------------------------------------------------------------------

# Patterns that signal the query is scoped to a specific source modality.
# When matched, chunks from other modalities are filtered out before reranking
# so the reranker never scores a PDF chunk for a "what does the video say…" query.
_VIDEO_SCOPE_RE = re.compile(
    r"\b(video|presenter|says in the video|according to the video|in the video)\b",
    re.IGNORECASE,
)
_IMAGE_SCOPE_RE = re.compile(
    r"\b(diagram|architecture diagram|figure|image|in the (image|diagram|figure)|"
    r"what (does|is) (shown|labeled|depicted) in)\b",
    re.IGNORECASE,
)
# Text/paper scope — used only to detect cross-modal queries that reference
# the written paper alongside a video or image source.  We do NOT filter to
# text-only chunks on a text-scope match: BM25 already ranks PDF chunks highly
# for "paper" queries and an aggressive filter risks excluding the specific PDF
# chunks needed when the reranker scores them just below the threshold.
_TEXT_SCOPE_RE = re.compile(
    r"\b(paper|the paper|document|report|according to the paper)\b",
    re.IGNORECASE,
)

_VIDEO_MODALITIES: frozenset[str] = frozenset({"video_summary", "video_clip"})
_IMAGE_MODALITIES_SET: frozenset[str] = frozenset({"image_global", "image_local"})


def _filter_by_modality_scope(query: str, chunks: list[dict]) -> list[dict]:
    """Drop chunks whose modality doesn't match an explicit source scope in the query.

    If the query says "according to the video …" we remove PDF and image chunks
    before reranking — they would score high on topical overlap but the judge
    (and the user) only want video-sourced evidence.

    Falls back to the full list when:
      - No scope keywords are detected.
      - BOTH video AND image scope keywords are detected (cross-modal query that
        requires evidence from multiple modalities — filtering either source type
        would prevent the model from synthesising across them).
      - Filtering would remove ALL chunks (safety guard).
    """
    has_video_scope = bool(_VIDEO_SCOPE_RE.search(query))
    has_image_scope = bool(_IMAGE_SCOPE_RE.search(query))
    has_text_scope = bool(_TEXT_SCOPE_RE.search(query))

    # Cross-modal queries reference more than one source type simultaneously
    # (e.g. "compare the video's explanation to the architecture diagram", or
    # "how does the video's explanation compare to the paper's description").
    # Filtering any single source type would prevent cross-source synthesis.
    cross_modal = (
        (has_video_scope and has_image_scope)
        or (has_video_scope and has_text_scope)
        or (has_image_scope and has_text_scope)
    )
    if cross_modal:
        logger.info(
            "modality-routing: cross-modal query detected "
            f"(video={has_video_scope} image={has_image_scope} text={has_text_scope}) — "
            "keeping full candidate list for cross-source synthesis"
        )
        return chunks

    if has_video_scope:
        filtered = [c for c in chunks if c.get("modality", "") in _VIDEO_MODALITIES]
        if filtered:
            logger.info(
                f"modality-routing: video scope detected — "
                f"kept {len(filtered)}/{len(chunks)} video chunks"
            )
            return filtered
        logger.warning(
            "modality-routing: video scope detected but no video chunks available — "
            "keeping full candidate list"
        )

    elif has_image_scope:
        filtered = [c for c in chunks if c.get("modality", "") in _IMAGE_MODALITIES_SET]
        if filtered:
            logger.info(
                f"modality-routing: image/diagram scope detected — "
                f"kept {len(filtered)}/{len(chunks)} image chunks"
            )
            return filtered
        logger.warning(
            "modality-routing: image scope detected but no image chunks available — "
            "keeping full candidate list"
        )

    return chunks


# ---------------------------------------------------------------------------
# Reranker helpers
# ---------------------------------------------------------------------------

_RERANK_THRESHOLD: float = 0.25
"""Minimum rerank score to keep a chunk.  Chunks below this are noise —
topically adjacent but don't contain the specific fact needed to answer the
query.
"""

_RERANK_MIN_FALLBACK: int = 3
"""Minimum number of chunks to keep when all candidates fall below
_RERANK_THRESHOLD.  Multi-hop reasoning queries often require several chunks
that each score modestly individually but together contain the full answer;
keeping at least 3 prevents the pipeline from starving the generator of
necessary context.
"""


def _build_rerank_parts(query: str, chunks: list[dict]) -> list[types.Part]:
    """Build an interleaved multimodal content list for a single Gemini rerank call.

    Each chunk gets its native representation:
      - text / pdf / video_summary → plain text (400-char truncation).
      - image_global / image_local → actual image bytes fetched from GCS.
      - video_clip                 → the pre-extracted mid-point JPEG frame
                                     from GCS plus the text summary as context.
                                     Using a frame avoids downloading the full
                                     MP4 clip at query time (~1000× smaller).
    If a GCS fetch fails for any chunk, that chunk falls back to its text
    representation so the reranker always receives a complete candidate list.

    All candidates are interleaved in one list of Parts so Gemini scores them
    relative to each other — the same calibration benefit as the original
    single-prompt design, extended to mixed modalities.
    """
    header = (
        "You are a relevance reranker for a retrieval-augmented generation system.\n\n"
        "Score each passage for how useful it is to directly answer the query.\n\n"
        "Respond with ONLY a JSON array — no explanation, no markdown fences:\n"
        '[{"id": <1-indexed int>, "score": <float 0.0–1.0>}, ...]\n\n'
        "Scoring rubric:\n"
        "  1.0 = passage explicitly contains the specific fact(s) needed to answer the query\n"
        "  0.7 = passage contains most of the answer but is missing one key detail\n"
        "  0.5 = passage contains related facts that contribute to the answer but do not\n"
        "        answer it directly (e.g. provides context, partial data, or supporting info)\n"
        "  0.2 = passage mentions the topic but does not contain the specific answer\n"
        "        (e.g. architecture diagram that labels a component without explaining it)\n"
        "  0.0 = completely unrelated OR the query is scoped to a specific source\n"
        "        (e.g. 'in the video', 'in the diagram') and this passage is from a\n"
        "        different source type\n\n"
        f"Query: {query}\n\nPassages:\n"
    )
    # The Gemini SDK accepts a mixed list of strings and Part objects, so plain
    # strings are used for text segments — no Part.from_text() wrapper needed.
    parts: list = [header]

    for i, chunk in enumerate(chunks):
        modality: str = chunk.get("modality", "")
        gcs_uri: str = chunk.get("gcs_uri", "")
        label = f"[{i + 1}]"

        if modality in ("image_global", "image_local") and gcs_uri:
            try:
                img_bytes = gcs.download_bytes(gcs_uri)
                mime = "image/png" if gcs_uri.endswith(".png") else "image/jpeg"
                parts.append(f"{label} ")
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
                continue
            except Exception as exc:
                logger.warning(
                    f"rerank: GCS fetch failed for image chunk {i+1} "
                    f"({gcs_uri}): {exc} — falling back to text"
                )

        elif modality == "video_clip" and gcs_uri:
            try:
                frame_bytes = gcs.download_bytes(gcs_uri)
                # Include the text summary so the model has both visual and
                # semantic context for the clip.
                context = chunk.get("text", "")[:300]
                parts.append(f"{label} [Video segment] Summary: {context}\nFrame: ")
                parts.append(types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg"))
                continue
            except Exception as exc:
                logger.warning(
                    f"rerank: GCS fetch failed for video frame chunk {i+1} "
                    f"({gcs_uri}): {exc} — falling back to text"
                )

        # Fallback path: text (used for pdf/text chunks, video_summary chunks,
        # and any media chunk whose GCS fetch failed above).
        parts.append(f"{label} {chunk['text'][:400]}")

    return parts


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Score all candidate chunks in a single multimodal Gemini Flash call, keep top_k.

    Sends one request containing all chunks — text as plain text, images as their
    actual bytes, video clips as a pre-extracted mid-point JPEG frame —
    so the model scores candidates relative to each other in a single calibrated
    pass.  Falls back to the existing RRF order if the API call or JSON parsing
    fails.

    Pre-processing:
      - Modality routing: if the query is scoped to a specific source (e.g.
        "according to the video"), chunks from other modalities are dropped
        before the Gemini call so the reranker only sees relevant candidates.

    Post-processing:
      - Chunks below _RERANK_THRESHOLD are dropped as noise (topically related
        but not answer-bearing).  At least 1 chunk is always kept.

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

    # Fix 2: filter to the query's target modality before reranking
    candidates_for_rerank = _filter_by_modality_scope(query, chunks)

    # Only skip the reranker when there is literally nothing to rank.
    # The old guard was `<= top_k`, which fired when the modality filter
    # reduced the pool to exactly top_k candidates — bypassing Gemini scoring
    # and leaving chunks in raw RRF order, hurting context precision.
    if len(candidates_for_rerank) <= 1:
        for c in candidates_for_rerank:
            c["rerank_score"] = c.get("score", 0.0)
        return candidates_for_rerank

    rerank_parts = _build_rerank_parts(query, candidates_for_rerank)

    try:
        response = _client.models.generate_content(
            model=config.RERANK_MODEL,
            contents=rerank_parts,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        if response.text is None:
            raise ValueError("Reranker returned empty response (possible safety filter)")
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
        for c in candidates_for_rerank[:top_k]:
            c["rerank_score"] = c.get("score", 0.0)
        return candidates_for_rerank[:top_k]

    for i, chunk in enumerate(candidates_for_rerank):
        chunk["rerank_score"] = id_to_score.get(i + 1, 0.0)

    ranked = sorted(candidates_for_rerank, key=lambda c: c["rerank_score"], reverse=True)

    # Drop chunks below _RERANK_THRESHOLD — they are topically adjacent but
    # do not contain the specific answer.  The threshold (0.25) filters noise
    # while _RERANK_MIN_FALLBACK ensures multi-hop queries always receive
    # enough context: individual chunks for multi-hop questions often score
    # modestly alone but are collectively sufficient to answer the question.
    top_candidates = ranked[:top_k]
    filtered = [c for c in top_candidates if c["rerank_score"] >= _RERANK_THRESHOLD]
    if not filtered:
        fallback_n = min(_RERANK_MIN_FALLBACK, len(top_candidates))
        filtered = top_candidates[:fallback_n]
        logger.warning(
            f"rerank: all chunks below threshold {_RERANK_THRESHOLD} — "
            f"keeping top-{fallback_n} as fallback context"
        )

    logger.info(
        f"rerank: {len(filtered)}/{len(top_candidates)} chunks kept "
        f"(threshold={_RERANK_THRESHOLD}, "
        f"scores: {[round(c['rerank_score'], 3) for c in filtered]})"
    )
    for i, c in enumerate(filtered):
        logger.debug(
            f"  [{i + 1}] rerank={c['rerank_score']:.3f} rrf={c.get('score', 0):.4f} "
            f"src={c.get('source', '?')} pg={c.get('page', '?')} "
            f"| {c['text'][:100]!r}"
        )
    return filtered
