# Multimodal Notebook + Eval Dashboard — Claude Code Context

## What This Project Is

A production-grade Retrieval-Augmented Generation (RAG) system.
Users upload documents, images, and videos. They can then ask
natural language questions about them. The system retrieves
relevant content using hybrid search, reranks results, and
generates grounded answers using Gemini 1.5 Pro.

A second tab in the UI shows an Eval Dashboard that runs a
golden dataset of 50 test queries and tracks quality metrics
over time using an LLM-as-a-judge approach.

## Portfolio Purpose

This project is being built to demonstrate GenAI engineering
knowledge. It must reflect
production-grade thinking: proper retrieval pipelines, eval
frameworks, and cost-aware architecture decisions.

---

## Architecture (Read This Before Writing Any Code)

### Ingestion Pipeline

1. User uploads file → saved to GCS
2. PDF → split into 6-page chunks → each chunk embedded directly as PDF
3. Images → passed whole to Gemini Embedding 2 (no change)
4. Video → split into 128s segments → each segment embedded directly as video/mp4
5. Audio → passed directly to Gemini Embedding 2 (max 80s per file)
6. All chunks → Gemini Embedding 2 → stored in ChromaDB
7. BM25 index built from raw text chunks in parallel

### Retrieval Pipeline (Hybrid Search + Reranking)

1. User query → HyDE (generate hypothetical answer, embed that)
2. BM25 keyword search → top 20 results
3. Vector search (ChromaDB cosine similarity) → top 20 results
4. Reciprocal Rank Fusion (RRF) → merge into single ranked list
5. Reranker (Gemini Flash) → score top 20 → keep top 5
6. Dynamic context window check → summarize if >80% full
7. Gemini 1.5 Pro generates answer with Chain-of-Thought prompt

### Eval Dashboard

- Golden dataset: 50 hard queries with ground truth answers
- LLM-as-judge: Gemini 1.5 Pro scores each response 0-5
- Metrics tracked: correctness, hallucination rate, faithfulness,
context precision, latency (p50/p95), token usage
- Results stored in SQLite, visualized in React with Recharts
- Before/after prompt comparison table

---

## File Structure

```
/backend
  main.py                  — FastAPI app, CORS, router registration
  config.py                — All env vars loaded here, nowhere else
  /routers
    upload.py              — POST /upload (file ingestion pipeline)
    query.py               — POST /query (RAG answer generation)
    eval.py                — POST /eval/run, GET /eval/results
  /services
    chunking.py            — PDF/image/video → chunk dicts
    embeddings.py          — Gemini Embedding 2 calls
    vectorstore.py         — ChromaDB add/search operations
    bm25_index.py          — BM25 build and search
    retrieval.py           — HyDE, RRF fusion, reranking logic
    generation.py          — Gemini 1.5 Pro prompt builder + call
    context_manager.py     — Dynamic context window management
  /eval
    judge.py               — LLM-as-judge scoring logic
    runner.py              — Batch eval pipeline
    golden_dataset.json    — 50 test queries with ground truth
    eval_results.db        — SQLite results store
  /models
    schemas.py             — All Pydantic models live here
/frontend
  /src
    /components
      FileUpload.jsx        — Drag and drop upload component
      ChatInterface.jsx     — Query input + answer display
      EvalDashboard.jsx     — Metrics table + charts
    App.jsx
    main.jsx
```

---

## Immutable Technical Decisions

**Do NOT change these without explicit discussion:**

- **Embedding model:** `gemini-embedding-2-preview` (Gemini Embedding 2)
— chosen because it supports text, image, and video natively
- **Vector DB:** ChromaDB with cosine similarity (local persistence)
— do NOT switch to Pinecone or Weaviate
- **LLM:** `gemini-1.5-pro` for generation and judging
- **Reranker:** `gemini-1.5-flash` (cheaper, fast enough for reranking)
- **Image region detection:** `gemini-3.1-flash-lite-preview` (cost-efficient multimodal, used in `chunk_image`)
- **Retrieval:** Hybrid BM25 + vector with RRF fusion
— do NOT use vector-only retrieval
- **Chunking:** 512 tokens, 64 overlap, via PyMuPDF
— do NOT use LangChain's text splitter
- **PDF parsing:** PyMuPDF (fitz) — do NOT use pdfplumber or pypdf
- **Orchestration:** Build retrieval pipeline manually
— do NOT use LangChain retrievers or LlamaIndex
— LangChain is allowed ONLY for RecursiveCharacterTextSplitter

---

## Coding Standards (Enforce on Every File)

### Python

- All functions must have type hints on parameters and return values
- Every service function must have a docstring (one line minimum)
- Use specific exception types — never bare `except:`
- No hardcoded strings — all config comes from `config.py`
- Use `loguru` for logging, not `print()`
- Async functions for all I/O (FastAPI endpoints, GCS calls)
- Pydantic models for all request/response shapes in `schemas.py`

### React / Frontend

- Functional components only, no class components
- TailwindCSS for all styling — no inline styles
- No component should exceed 150 lines — split if larger
- All API calls go through a `/src/api/` module, never inline fetch

### General

- Every PR-sized chunk of work gets a commit with a clear message
- Format: `feat:`, `fix:`, `refactor:`, `test:`, `docs:` prefixes
- No commented-out code — delete it or use a TODO comment

---

## Key Functions — Name These Exactly

When implementing, use these exact function signatures so
nothing breaks when files reference each other:

```python
# chunking.py
def chunk_pdf(filepath: str) -> list[dict]: ...
def chunk_image(filepath: str) -> list[dict]: ...
def chunk_video(filepath: str) -> list[dict]: ...

# embeddings.py
def embed_chunks(chunks: list[dict]) -> list[dict]: ...
def embed_text(text: str) -> list[float]: ...

# vectorstore.py
def add_chunks(chunks: list[dict]) -> None: ...
def search(query_embedding: list[float], top_k: int = 20) -> list[dict]: ...

# bm25_index.py
def build_index(chunks: list[dict]) -> None: ...
def search_bm25(query: str, top_k: int = 20) -> list[dict]: ...

# retrieval.py
def hybrid_search(query: str, top_k: int = 20) -> list[dict]: ...
def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]: ...
def hyde_expand(query: str) -> str: ...
def reciprocal_rank_fusion(bm25_results, vector_results) -> list[dict]: ...

# generation.py
def generate_answer(question: str, chunks: list[dict]) -> dict: ...
```

---

## What NOT to Build (Scope Boundaries)

- No user authentication — single user, no login
- No multi-tenancy — one shared ChromaDB collection
- No real-time streaming of answers (Week 3 stretch goal only)
- No fine-tuning — inference only
- No Vertex AI Agent Builder — build all pipelines manually
- No Docker until Week 3 — run locally first

---

## Current Build Status

Update this section as you complete each step:

- Week 1: Baseline RAG (upload → chunk → embed → query)
- Week 2: Hybrid search + reranking + advanced prompting
- Week 3: Eval dashboard + golden dataset + polish

---

## Common Mistakes to Avoid

- Do not store embeddings in ChromaDB AND a separate file — ChromaDB is the single source of truth
- Do not call the Gemini embedding API one chunk at a time — batch documents in groups of 20 (20k token/request limit; max 250 inputs/request — see https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings)
- Do not re-embed a file that's already indexed — check SHA-256 hash first
- Do not send all 20 retrieved chunks to the LLM — rerank to 5 first
- Do not build the BM25 index in the query path — build it during ingestion, load it at startup

