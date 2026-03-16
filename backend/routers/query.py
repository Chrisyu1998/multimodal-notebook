"""
Query router — POST /query
Accepts a natural language question, runs the full RAG retrieval pipeline,
and returns a grounded answer from Gemini 1.5 Pro.

Pipeline order:
  1. HyDE — expand query into hypothetical answer, embed it
  2. BM25 keyword search  → top 20
  3. Vector search (ChromaDB) → top 20
  4. Reciprocal Rank Fusion → merged ranked list
  5. Rerank with Gemini Flash → top 5
  6. Dynamic context window check → summarize if >80% full
  7. Gemini 1.5 Pro generates answer (Chain-of-Thought prompt)
"""

import time
from fastapi import APIRouter, HTTPException
from loguru import logger

import backend.config as config
from backend.models.schemas import QueryRequest, QueryResponse, SourceReference
from backend.services import embeddings, vectorstore, generation

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Embed question → vector search → generate grounded answer with Gemini."""
    logger.info(f"Query received: {request.question!r}")
    start = time.monotonic()

    try:
        query_embedding = embeddings.embed_text(request.question)
    except Exception as exc:
        logger.error(f"Embedding failed: {exc}")
        raise HTTPException(status_code=502, detail="Failed to embed question") from exc

    try:
        chunks = vectorstore.search_balanced(query_embedding)[:5]
    except Exception as exc:
        logger.error(f"Vector search failed: {exc}")
        raise HTTPException(status_code=502, detail="Vector search failed") from exc

    if not chunks:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.warning(f"No chunks found for query in {elapsed_ms:.1f}ms")
        return QueryResponse(
            answer="I don't have enough information to answer this.",
            sources=[],
            chunks_used=0,
            model=config.GENERATION_MODEL,
        )

    try:
        result = generation.generate_answer(request.question, chunks)
    except Exception as exc:
        logger.error(f"Generation failed: {exc}")
        raise HTTPException(status_code=502, detail="Answer generation failed") from exc

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(f"Query completed in {elapsed_ms:.1f}ms — {result['chunks_used']} chunks used")

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceReference(**s) for s in result["sources"]],
        chunks_used=result["chunks_used"],
        model=result["model"],
    )
