"""
Query router — POST /query
Accepts a natural language question, runs the full RAG retrieval pipeline,
and returns a grounded answer from Gemini 2.5 Flash.

Pipeline order:
  1. Guard: reject if no documents are indexed yet
  2. HyDE — expand query into hypothetical answer, embed it
  3. BM25 keyword search  → top 20
  4. Vector search (ChromaDB) → top 20
  5. Reciprocal Rank Fusion → merged ranked list
  6. Rerank with Gemini Flash → top 5
  7. Dynamic context window check → summarize if >80% full
  8. Gemini 2.5 Flash generates answer (Chain-of-Thought prompt)
"""

import time
from fastapi import APIRouter, HTTPException
from loguru import logger

import backend.config as config
from backend.models.schemas import QueryRequest, QueryResponse, SourceReference
from backend.services import embeddings, vectorstore, generation
from backend.services.embeddings import EmbeddingBatchError
from backend.services.generation import GenerationError
from backend.services.vectorstore import VectorStoreUnavailableError

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Embed question → vector search → generate grounded answer with Gemini."""
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

    # ---- 2. Embed question ----
    try:
        query_embedding = embeddings.embed_text(request.question)
    except EmbeddingBatchError as exc:
        logger.error(f"Embedding failed for query: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Embedding service unavailable. Check your API key.",
        ) from exc
    except Exception as exc:
        logger.error(f"Unexpected embedding error: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Embedding service unavailable. Check your API key.",
        ) from exc

    # ---- 3. Retrieve chunks ----
    try:
        chunks = vectorstore.search_balanced(query_embedding)[:5]
    except VectorStoreUnavailableError as exc:
        logger.error(f"Vector store unavailable during search: {exc}")
        raise HTTPException(status_code=503, detail="Vector store unavailable.") from exc
    except Exception as exc:
        logger.error(f"Vector search failed: {exc}")
        raise HTTPException(status_code=502, detail="Vector search failed.") from exc

    if not chunks:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.warning(f"No chunks matched query in {elapsed_ms:.1f}ms")
        return QueryResponse(
            answer="I don't have enough information to answer this.",
            sources=[],
            chunks_used=0,
            model=config.GENERATION_MODEL,
        )

    # ---- 4. Generate answer ----
    try:
        result = generation.generate_answer(request.question, chunks)
    except GenerationError as exc:
        logger.error(f"Generation failed: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Embedding service unavailable. Check your API key.",
        ) from exc
    except Exception as exc:
        logger.error(f"Unexpected generation error: {exc}")
        raise HTTPException(status_code=502, detail="Answer generation failed.") from exc

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(f"Query completed in {elapsed_ms:.1f}ms — {result['chunks_used']} chunks used")

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceReference(**s) for s in result["sources"]],
        chunks_used=result["chunks_used"],
        model=result["model"],
    )
