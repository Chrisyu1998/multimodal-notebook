"""
All Pydantic request/response models for the API.
No business logic lives here — shapes only.
"""

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    """Returned immediately after a file is saved to disk (pre-ingestion)."""
    file_id: str
    filename: str
    size_bytes: int
    status: str
    num_chunks: int = 0


class IngestResponse(BaseModel):
    """Returned after the full ingestion pipeline completes (Week 1)."""
    file_id: str
    filename: str
    num_chunks: int
    message: str


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Payload for POST /query."""
    question: str


class SourceReference(BaseModel):
    """A single source chunk surfaced alongside the answer."""
    filename: str
    page: int
    score: float
    snippet: str = ""


class QueryResponse(BaseModel):
    """Returned after the RAG pipeline generates an answer."""
    answer: str
    sources: list[SourceReference]
    chunks_used: int
    model: str
    media_chunks_degraded: int = 0


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

from typing import Optional


class EvalRunMeta(BaseModel):
    """Lightweight summary row for one eval run (from SQLite)."""

    run_id: str
    timestamp: str
    dataset_version: str
    num_queries: int
    avg_latency_ms: float
    p95_latency_ms: float
    failed_queries: int
    avg_correctness: float
    avg_hallucination_rate: float
    avg_faithfulness: float
    avg_context_precision: float


class EvalQueryResult(BaseModel):
    """Per-query result from a single eval run."""

    query_id: str
    query: str
    category: str
    source_modality: str
    status: str
    generated_answer: str
    ground_truth: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    scores: Optional[dict] = None
    reasoning: Optional[dict] = None


class EvalRunDetail(BaseModel):
    """Full eval run payload — summary + per-query results."""

    run_id: str
    timestamp: str
    dataset_version: str
    results: list[EvalQueryResult]
    summary: dict
