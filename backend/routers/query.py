"""
Query router — POST /query
Accepts a natural language question, runs the full RAG retrieval pipeline,
and returns a grounded answer from Gemini 2.5 Flash.

Pipeline order:
  1. Guard: reject if no documents are indexed yet
  2. hybrid_search — HyDE + BM25 + vector + RRF fusion → top 20 chunks
  3. rerank        — one Gemini Flash call scores all 20 chunks → top 5
  4. Gemini 2.5 Flash generates answer (grounded, Chain-of-Thought prompt)
"""

import time
from fastapi import APIRouter, HTTPException
from loguru import logger

import backend.config as config
from backend.models.schemas import QueryRequest, QueryResponse, SourceReference
from backend.services import generation, vectorstore
from backend.services.embeddings import EmbeddingBatchError
from backend.services.generation import (
    GenerationError,
    GenerationConfigError,
    GenerationRetryableError,
)
from backend.services.retrieval import hybrid_search, rerank
from backend.services.vectorstore import VectorStoreUnavailableError

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Run hybrid retrieval → generate grounded answer with Gemini."""
    logger.info(f"Query received: {request.question!r}")
    start = time.monotonic()

    # ---- 1. Guard: nothing indexed yet ----
    try:
        if vectorstore.collection_is_empty():
            logger.warning("Query rejected — no documents indexed yet.")
            raise HTTPException(
                status_code=400,
                detail="No documents indexed yet. Please upload a file first.",
            )
    except VectorStoreUnavailableError as exc:
        logger.error(f"Vector store unavailable during empty-collection check: {exc}")
        raise HTTPException(status_code=503, detail="Vector store unavailable.") from exc

    # ---- 2. Hybrid search: HyDE + BM25 + vector + RRF → top 20 ----
    try:
        chunks = hybrid_search(request.question, top_k=config.VECTOR_TOP_K)
    except EmbeddingBatchError as exc:
        logger.error(f"Embedding failed during hybrid search: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Embedding service unavailable. Check your API key.",
        ) from exc
    except VectorStoreUnavailableError as exc:
        logger.error(f"Vector store unavailable during hybrid search: {exc}")
        raise HTTPException(status_code=503, detail="Vector store unavailable.") from exc
    except Exception as exc:
        logger.error(f"Hybrid search failed: {exc}")
        raise HTTPException(status_code=502, detail="Retrieval failed.") from exc

    if not chunks:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.warning(f"No chunks matched query in {elapsed_ms:.1f}ms")
        return QueryResponse(
            answer="I don't have enough information to answer this.",
            sources=[],
            chunks_used=0,
            model=config.GENERATION_MODEL,
        )

    # ---- 3. Rerank: one Gemini Flash call scores all 20 chunks, keep top 5 ----
    reranked_chunks = rerank(request.question, chunks, top_k=config.RERANK_TOP_K)

    try:
        result = generation.generate_answer(request.question, reranked_chunks)
    except GenerationConfigError as exc:
        logger.error(f"Generation misconfiguration: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Generation service misconfigured. Contact the administrator.",
        ) from exc
    except GenerationRetryableError as exc:
        logger.error(f"Generation transient failure: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Generation service unavailable. Please try again.",
        ) from exc
    except GenerationError as exc:
        logger.error(f"Generation failed: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Generation service unavailable. Check your API key.",
        ) from exc
    except Exception as exc:
        logger.error(f"Unexpected generation error: {exc}")
        raise HTTPException(status_code=502, detail="Answer generation failed.") from exc

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        f"Query completed in {elapsed_ms:.1f}ms — {result['chunks_used']} chunks used"
        + (f", {result['media_chunks_degraded']} media chunk(s) degraded" if result["media_chunks_degraded"] else "")
    )

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceReference(**s) for s in result["sources"]],
        chunks_used=result["chunks_used"],
        model=result["model"],
        media_chunks_degraded=result["media_chunks_degraded"],
    )
