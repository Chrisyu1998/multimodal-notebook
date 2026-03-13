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


class ChunkReference(BaseModel):
    """A single retrieved chunk surfaced alongside the answer."""
    chunk_id: str
    text: str
    source: str
    score: float


class QueryResponse(BaseModel):
    """Returned after the RAG pipeline generates an answer."""
    answer: str
    chunks_used: list[ChunkReference]
    latency_ms: float


# ---------------------------------------------------------------------------
# Eval  (stubs — filled out in Week 3)
# ---------------------------------------------------------------------------

class EvalRunRequest(BaseModel):
    """Payload for POST /eval/run."""
    # TODO (Week 3): add run config options (subset size, prompt variant, etc.)
    pass


class EvalResult(BaseModel):
    """Single row from the eval results store."""
    # TODO (Week 3): define fields (query_id, score, latency, tokens, etc.)
    pass


class EvalRunResponse(BaseModel):
    """Summary returned after an eval run completes."""
    # TODO (Week 3): aggregate metrics, run_id, timestamp
    pass
