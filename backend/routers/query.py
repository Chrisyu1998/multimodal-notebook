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

from backend.models.schemas import QueryRequest, QueryResponse

# TODO: import services once implemented
# from backend.services import retrieval, generation

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Run the full RAG pipeline for a user question and return an answer."""
    logger.info(f"Query received: {request.question!r}")
    start = time.monotonic()

    # TODO: retrieval.hybrid_search(request.question)  → chunks
    # TODO: retrieval.rerank(request.question, chunks)  → top 5
    # TODO: generation.generate_answer(request.question, top_chunks)

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(f"Query completed in {elapsed_ms:.1f}ms")

    raise HTTPException(status_code=501, detail="Not implemented yet")
