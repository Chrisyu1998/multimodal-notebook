# Product Requirements Document: Multimodal RAG Notebook

**Author:** Chris Yu
**Last Updated:** 2026-03-25
**Status:** Living Document

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals and Non-Goals](#3-goals-and-non-goals)
4. [User Persona](#4-user-persona)
5. [System Architecture Overview](#5-system-architecture-overview)
6. [Feature Specifications](#6-feature-specifications)
7. [Design Decisions with Trade-off Analysis](#7-design-decisions-with-trade-off-analysis)
8. [Technology Stack](#8-technology-stack)
9. [Data Models and API Surface](#9-data-models-and-api-surface)
10. [Evaluation Framework](#10-evaluation-framework)
11. [Known Limitations and Future Work](#11-known-limitations-and-future-work)
12. [Success Criteria](#12-success-criteria)

---

## 1. Executive Summary

Multimodal RAG Notebook is a production-grade Retrieval-Augmented Generation system that accepts PDFs, images, and videos and answers natural language questions about them. It combines hybrid BM25 + vector search, LLM reranking, and grounded answer generation via Gemini 2.5 Flash into a single pipeline — built from scratch without framework orchestration (no LangChain retrievers, no LlamaIndex).

A companion Eval Dashboard runs a golden dataset of 51 test queries and tracks correctness, hallucination rate, faithfulness, and context precision over time using an LLM-as-a-judge approach.

The project exists to demonstrate GenAI engineering competency at a portfolio level: proper retrieval pipelines, cost-aware model selection, multimodal media handling, and a measurable eval framework.

---

## 2. Problem Statement

### The Gap

Most RAG tutorials and portfolio projects stop at "embed text, cosine search, feed to LLM." This leaves several critical production concerns unaddressed:

- **Multimodal content** — real knowledge bases contain diagrams, charts, and videos, not just text. A text-only pipeline cannot answer "what does the architecture diagram show?"
- **Retrieval quality** — vector-only search fails on exact-match queries (model names, version numbers, acronyms). There is no mechanism to catch these misses.
- **Answer grounding** — without citation enforcement, the LLM freely hallucinated from parametric knowledge. Users have no way to verify claims.
- **Eval gap** — without a structured evaluation framework, prompt changes and retrieval tuning are done blind. There is no feedback signal on whether a change helped or hurt.

### What This Project Proves

That a single engineer can build a complete, instrumentable RAG system that handles multiple modalities, uses hybrid retrieval with reranking, enforces grounded generation with inline citations, and measures its own quality via an automated eval suite — all without leaning on opaque framework abstractions.

---

## 3. Goals and Non-Goals

### Goals

| # | Goal | Measurable Outcome |
|---|------|-------------------|
| G1 | Accept PDFs, images, and videos as knowledge sources | All three modalities upload, chunk, embed, and index without error |
| G2 | Answer natural language questions grounded in uploaded content | Answers include inline `[N]` citations traceable to source chunks |
| G3 | Hybrid retrieval that catches both semantic and exact-match queries | BM25 + vector fusion outperforms vector-only on the golden dataset |
| G4 | Automated quality measurement | Eval dashboard reports correctness, hallucination, faithfulness, context precision per run |
| G5 | Cost-aware architecture | Total generation cost per query stays under $0.01 USD at current Gemini pricing |
| G6 | No framework orchestration | All retrieval, fusion, reranking, and generation logic is hand-built and debuggable |

### Non-Goals

- **User authentication or multi-tenancy** — single user, one shared collection
- **Real-time streaming** — answers return only when generation completes
- **Fine-tuning** — inference only, no model training
- **Managed vector DB** — ChromaDB local persistence, no Pinecone/Weaviate
- **Docker / containerization** — runs locally in dev
- **Automated eval feedback loop** — eval scores are displayed, not used to auto-tune the pipeline

---

## 4. User Persona

**Primary:** An interviewer or hiring manager evaluating GenAI engineering depth. They will:

1. Upload a sample PDF, image, or video
2. Ask factual, multi-hop, and cross-modal questions
3. Inspect source citations in the answer
4. Check the Eval Dashboard for quality metrics
5. Read the codebase for architectural clarity

**Secondary:** The developer (Chris) iterating on retrieval quality. They will:

1. Run the eval suite after prompt or pipeline changes
2. Compare before/after metrics in the dashboard
3. Drill into per-query results to identify failure modes

---

## 5. System Architecture Overview

### Ingestion Pipeline

```
File Upload → Validation → SHA-256 Dedup Check
    │
    ▼
Modality-Specific Chunking
├── PDF  → 800-token / 100-overlap, TOC-aware boundaries
├── Image → Global (full image + caption) + per-region crops (Gemini ROI)
└── Video → Scene detection (PySceneDetect) → dual-stream (MP4 clip + text summary)
    │
    ▼
Binary Media (images, video frames) → Google Cloud Storage (gs:// URIs stamped on chunks)
    │
    ▼
Gemini Embedding 2 (batched: text=20, image=6, video=1 per request)
    │
    ├──▶ ChromaDB (cosine similarity, persistent)
    └──▶ BM25 Index (Porter stemmer, pickled to disk)
```

### Query Pipeline

```
User Question
    │
    ▼
HyDE Expansion (Gemini Flash generates hypothetical answer → embed that)
    │
    ├──▶ BM25 keyword search → top 20
    └──▶ Vector search (ChromaDB) → top 20
            │
            ▼
    Reciprocal Rank Fusion (RRF, k=60)
    → 20 unique merged candidates
            │
            ▼
    Modality Routing (filter by video/image/text scope if detected)
            │
            ▼
    LLM Reranker (Gemini Flash, single multimodal call)
    → top 5 with score ≥ 0.25 (minimum 3 fallback)
            │
            ▼
    Grounded Generation (Gemini 2.5 Flash, CoT thinking, 1024 token budget)
    → Answer with inline [N] citations + source metadata
```

### Eval Pipeline

```
Golden Dataset (51 queries) → Concurrent RAG pipelines (semaphore=2)
    │
    ▼
LLM-as-Judge (Gemini 2.5 Pro) scores each response on 4 dimensions
    │
    ▼
JSON results file + SQLite metadata → Eval Dashboard (React + Recharts)
```

---

## 6. Feature Specifications

### 6.1 File Upload and Ingestion

**Endpoint:** `POST /upload`
**Accepted formats:** PDF, PNG, JPEG, WebP, MP4, MOV
**Size limit:** 500 MB (configurable)
**Deduplication:** SHA-256 hash check — already-indexed files are rejected

The upload path validates the file, chunks it by modality, uploads binary media to GCS, embeds all chunks via Gemini Embedding 2, indexes into ChromaDB, and appends to the BM25 index.

### 6.2 Question Answering

**Endpoint:** `POST /query`
**Input:** `{ "question": "..." }`
**Output:** Answer text with `[N]` citations, source references (filename, page, score, snippet), operational metrics

The query path runs HyDE expansion, hybrid search, RRF fusion, modality routing, reranking, and grounded generation. Telemetry (latency, tokens) is logged to SQLite.

### 6.3 Eval Dashboard

**Endpoints:** `GET /eval/runs`, `GET /eval/latest`, `GET /eval/runs/{run_id}`
**UI tabs:** Metrics (per-query table), Comparison (before/after), System Metrics (latency + cost charts)
**Auto-refresh:** 60-second polling

### 6.4 System Metrics

**Endpoints:** `GET /metrics/timeseries`, `GET /metrics/summary`
**Metrics:** Latency (avg, p50, p95), token usage, estimated cost USD
**Cost formula:** Input $0.00000125/token, Output $0.000005/token

---

## 7. Design Decisions with Trade-off Analysis

### 7.1 Chunking Strategy: Custom PyMuPDF over Gemini-Native Document Processing

**Decision:** Build a custom chunker using PyMuPDF (fitz) with 800-token windows, 100-token overlap, and TOC-aware section boundaries. Do NOT use Gemini's document understanding to let the model "read" full PDFs.

| | Pros | Cons |
|---|------|------|
| **Custom PyMuPDF chunker** | Full control over chunk boundaries — can enforce token ceilings and section-aware splits. Deterministic and reproducible. No API cost for chunking. Overlap prevents losing information at boundaries. Junk token filtering catches OCR artifacts. | Requires maintenance: must handle edge cases (mega-paragraphs, malformed TOCs, multi-column layouts). Doesn't understand semantic coherence the way an LLM would. |
| **Gemini full-document ingestion** | Zero chunking code. Model "sees" the whole document. Better at complex layouts (tables, multi-column). | No control over what the model attends to. Non-deterministic token usage. Expensive at ingestion time for large PDFs. Cannot index individual chunks into a vector DB for retrieval — defeats the RAG pattern. |
| **LangChain RecursiveCharacterTextSplitter** | Battle-tested, handles many edge cases. | Splits on character count, not token count — unreliable with tiktoken. No TOC awareness. Adds a dependency for a simple operation. |

**Why this choice:** RAG requires individually-indexed chunks for retrieval. Passing full documents to Gemini at query time would be cost-prohibitive and bypass the retrieval pipeline entirely. PyMuPDF is the fastest Python PDF library, preserves layout metadata, and exposes the document TOC for section-boundary detection.

---

### 7.2 Dual-Stream Chunking for Images

**Decision:** Each image produces (a) one global chunk embedding the full image with a Gemini-generated caption, and (b) one chunk per region-of-interest (ROI) detected by Gemini Flash, cropped and captioned individually.

| | Pros | Cons |
|---|------|------|
| **Dual-stream (global + regions)** | A question about a specific chart matches the precise crop embedding, not the full-page average. Global chunk still handles broad "what's in this image" queries. Captions make images searchable by BM25. | More API calls at ingestion (ROI detection + per-crop captioning). More chunks in the index. Requires bbox validation and rescue logic for out-of-bounds coordinates. |
| **Global-only** | Simple. One chunk per image. | A dense page with 4 charts embeds as one point in vector space. A query about a specific chart competes against the full-page embedding — lower precision. |
| **Fixed grid tiling** | No LLM call needed. Deterministic. | Cuts across meaningful regions. A chart split between two tiles has no useful embedding in either half. |

**Why this choice:** The embedding of a full infographic with 5 regions is an average of everything — it matches nothing well. ROI detection produces semantically meaningful crops. The caption text enables BM25 to retrieve images by keyword, which pure-image embedding cannot.

---

### 7.3 Scene-Based Video Chunking with Dual-Stream

**Decision:** Use PySceneDetect for scene boundary detection, hard-split scenes over 120s (with 5s overlap), and produce two chunks per scene: native MP4 clip bytes (embedded as video) + text summary (embedded as text).

| | Pros | Cons |
|---|------|------|
| **Scene-based + dual-stream** | Cuts at visual transitions preserve semantic coherence. MP4 clip embedding captures motion and audio that text misses. Text summary enables BM25 retrieval. 5s overlap prevents hard cuts mid-sentence. | Requires PySceneDetect + ffmpeg as dependencies. 120s hard ceiling is a Gemini Embedding API constraint, not a semantic choice — some scenes must be force-split. Scene detection fails on slow-moving content (talking head, slideshows). |
| **Fixed-duration (e.g., 30s chunks)** | Simple, no scene detection needed. | Cuts mid-sentence, mid-action. No semantic coherence at boundaries. |
| **Transcript-only** | No video processing at all. Much cheaper. | Loses all visual information. Cannot answer "what does the slide at 2:30 show?" |
| **Full-video embedding** | Simplest approach. | Gemini Embedding API caps at 120s per input. Long videos require splitting anyway. A 30-minute video as a single embedding loses all temporal specificity. |

**Why this choice:** Scene detection approximates semantic boundaries better than fixed-duration cuts. The dual-stream approach (video + text) means the system can retrieve a scene by visual content (diagram on screen) OR by spoken keyword. The 120s ceiling is an API constraint — not negotiable.

---

### 7.4 Embedding Model: Gemini Embedding 2 over OpenAI / Cohere

**Decision:** Use `gemini-embedding-2-preview` as the single embedding model for all modalities.

| | Pros | Cons |
|---|------|------|
| **Gemini Embedding 2** | Only embedding model with native text + image + video support in one API. Single vector space — no per-modality index or routing logic. Cross-modal retrieval (text query → image result) works natively. | Preview-stage model — API surface may change. Batch limits differ by modality (text=20, image=6, video=1). Documentation is sparse. |
| **OpenAI text-embedding-3-large** | Best-in-class text embeddings. Mature, stable API. | Text-only. Would require a separate model for image/video embeddings and a strategy to align the two vector spaces. |
| **CLIP (open-source)** | Free, self-hosted. Text + image. | No video support. Requires GPU for reasonable latency. Embedding dimension (512) is lower than Gemini (768+). Adds infrastructure complexity. |
| **Per-modality models (e.g., text-embedding-3 + CLIP + VideoMAE)** | Best-of-breed per modality. | Three separate vector spaces. Cross-modal search requires multi-index queries and score normalization. Maintenance burden is 3x. |

**Why this choice:** The entire architecture premise is that a text query can retrieve an image or a video chunk. This requires a shared embedding space. Gemini Embedding 2 is the only production API that provides this today. The single-index design eliminates cross-modal routing complexity entirely.

---

### 7.5 Vector Store: ChromaDB over Pinecone / pgvector / Weaviate

**Decision:** Use ChromaDB with local persistent storage and cosine similarity.

| | Pros | Cons |
|---|------|------|
| **ChromaDB (local)** | Zero infrastructure — runs in-process. Persistent to disk. Cosine similarity out of the box. Simple Python API. No API keys, no network hops. Appropriate for portfolio-scale corpus (hundreds to low thousands of chunks). | Single-machine only. No horizontal scaling. No built-in replication or backup. At millions of chunks, memory becomes a bottleneck. No managed monitoring. |
| **Pinecone** | Managed, serverless. Scales to billions of vectors. Built-in metadata filtering and namespaces. | Adds cloud dependency and cost. Requires API key management. Network latency on every query. Overkill for a portfolio project. |
| **pgvector** | Familiar Postgres. Can co-locate with relational data. HNSW indexing available. | Requires running a Postgres instance. Vector indexing performance trails purpose-built vector DBs. ORM complexity. |
| **Weaviate** | Strong multimodal support. GraphQL API. Hybrid search built in. | Heavy runtime. Docker-based setup. More infrastructure than the RAG pipeline itself. |

**Why this choice:** The corpus is one technical report, its architecture diagram, and a video — hundreds of chunks, not millions. ChromaDB adds zero infrastructure overhead, runs in-process, and provides everything needed for cosine search with metadata filtering. Switching to a managed vector DB is a known upgrade path if scale demands it.

---

### 7.6 Hybrid Search (BM25 + Vector) with RRF over Vector-Only

**Decision:** Run BM25 and vector search in parallel, merge results with Reciprocal Rank Fusion (k=60), return top 20 candidates to the reranker.

| | Pros | Cons |
|---|------|------|
| **Hybrid BM25 + Vector + RRF** | BM25 catches exact-match queries (model names, d_model=512, BLEU scores) that vector search misses. RRF merges ranked lists without score normalization — critical because raw BM25 and cosine scores live on incomparable scales. Parameter-free fusion (no learned weights). Proven in IR literature (Cormack et al. 2009). | Two indexes to maintain. BM25 index loaded into memory at startup. Slightly more complex query path. BM25 only works on text chunks — video/image chunks without text are invisible to it. |
| **Vector-only** | Simpler. One index. Works on all modalities. | Fails on exact-match queries. "What is d_model in the base transformer?" retrieves chunks about model dimensions in general, not the specific parameter value. Acronyms and version numbers are averaged away during embedding. |
| **Weighted score combination** | Can tune weights per modality. | BM25 scores and cosine similarity scores are on different scales. Choosing weights requires a dev set and is fragile to corpus changes. RRF avoids this entirely by using rank order. |

**Why this choice:** The golden dataset includes factual queries that require exact parameter values. BM25 is essential for these. RRF is the standard fusion method because it operates on ranks (universally comparable) rather than scores (scale-dependent).

---

### 7.7 HyDE (Hypothetical Document Embedding) over Raw Query Embedding

**Decision:** Before vector search, generate a 2-3 sentence hypothetical answer to the query via Gemini Flash, then embed that hypothetical answer instead of the raw query.

| | Pros | Cons |
|---|------|------|
| **HyDE** | Bridges the question-answer embedding gap — a hypothetical answer is geometrically closer to real answer chunks than the bare question. Improves recall by 10-20% on long-form factual queries. Runs concurrently with BM25 tokenization, so no added latency on the critical path. | One extra Gemini Flash call per query. If the hypothetical answer is wrong, it can steer retrieval in the wrong direction. Adds complexity. |
| **Raw query embedding** | Simpler. No extra API call. | Questions and answers occupy different embedding regions. "What causes gradient vanishing?" is far from "Gradient vanishing occurs when..." in embedding space. Lower recall on factual queries. |
| **Query expansion (synonyms/rewriting)** | Can improve keyword coverage. | Doesn't address the question-answer distribution gap. BM25 already handles keyword matching. |

**Why this choice:** The retrieval pipeline's quality ceiling is set by recall — if the right chunk isn't in the top 20, no amount of reranking can save it. HyDE is the highest-impact recall improvement at the cost of one lightweight Flash call. The fallback to raw query embedding on HyDE failure ensures robustness.

---

### 7.8 LLM Reranker over Cross-Encoder / No Reranker

**Decision:** Send all 20 RRF candidates in a single multimodal Gemini Flash call. The model scores each chunk 0.0-1.0 for relevance to the query. Keep top 5 (threshold 0.25, minimum 3).

| | Pros | Cons |
|---|------|------|
| **LLM reranker (Gemini Flash)** | Multimodal — can score image bytes and video frames, not just text. Sees all 20 candidates simultaneously for relative comparison. Reuses the same model as generation — no additional API dependency. 0.25 threshold with 3-chunk minimum prevents over-filtering on hard queries. | One extra LLM call per query (~1-2s latency). Token cost scales with candidate count. Non-deterministic scoring. JSON response parsing requires fallback logic. |
| **Cross-encoder (e.g., ms-marco-MiniLM)** | Fast, deterministic. Lightweight model. | Text-only — cannot score image or video chunks. Requires hosting a model or adding an API dependency. Scores pairs independently, no relative comparison. |
| **No reranker (RRF order directly)** | Faster. No extra API call. | RRF order is based on retrieval rank, not semantic relevance to the specific query. Passing 20 chunks to generation wastes tokens and dilutes context quality. |
| **Cohere Rerank** | Purpose-built for reranking. Fast. | Text-only API. Adds another vendor dependency. Cannot score multimodal chunks. |

**Why this choice:** The corpus is multimodal. A text-only reranker cannot evaluate whether an architecture diagram is relevant to a query about attention mechanisms. Gemini Flash can — it scores image bytes and video frames alongside text chunks in a single call. The cost is one Flash inference per query, which is acceptable given the generation call already dominates latency.

---

### 7.9 Generation: Gemini 2.5 Flash over GPT-4o / Gemini 1.5 Pro / Claude

**Decision:** Use Gemini 2.5 Flash with Chain-of-Thought thinking (1024 token budget) for answer generation.

| | Pros | Cons |
|---|------|------|
| **Gemini 2.5 Flash** | ~3.3% hallucination rate (among lowest). Native multimodal — can see inline image bytes and video frames in context. 20x cheaper than 1.5 Pro. Native thinking mode (CoT budget) reduces unsupported claims. Same vendor as embeddings — one API key, one billing account. | Requires `temperature=1.0` when thinking mode is enabled (not configurable). Thinking tokens consume budget but are not visible in output. |
| **Gemini 1.5 Pro** | Higher quality ceiling on complex reasoning. | 20x more expensive. Marginal quality improvement for grounded generation where most reasoning is "read the context and cite." |
| **GPT-4o** | Strong general reasoning. Broad ecosystem. | Second API vendor (OpenAI). No native thinking mode. Would require prompt engineering for chain-of-thought. Higher hallucination rate than Gemini 2.5 Flash on grounded tasks. |
| **Claude 3.5 Sonnet** | Strong at following citation instructions. | Third API vendor. Not multimodal-native for image+video context in the same way. Adds vendor complexity for marginal benefit. |

**Why this choice:** Cost and grounding. For a RAG system where the LLM's primary job is to read 5 context chunks and cite them accurately, Gemini 2.5 Flash's combination of low hallucination rate, native multimodal context, built-in thinking mode, and 20x cost savings over Pro makes it the clear choice. Using the same vendor as embeddings keeps operational complexity minimal.

---

### 7.10 Eval: LLM-as-Judge over BLEU / ROUGE / Human Eval

**Decision:** Use Gemini 2.5 Pro as an automated judge scoring four dimensions: correctness (0-5), hallucination rate (0-1), faithfulness (0-5), and context precision (0-1).

| | Pros | Cons |
|---|------|------|
| **LLM-as-judge (4 dimensions)** | Evaluates semantic correctness, not just lexical overlap. Detects hallucinations that are fluent and well-written. Scores grounding in retrieved context (faithfulness) separately from factual accuracy (correctness). Automated — runs on every eval suite invocation. Structured rubrics constrain judge variance. | Non-deterministic — scores vary across runs. More expensive than BLEU/ROUGE. Judge model (2.5 Pro) is more expensive than generator (2.5 Flash). Requires careful prompt engineering for each scoring dimension. |
| **BLEU / ROUGE** | Deterministic. Free. Fast. | Measures n-gram overlap, not semantic correctness. Penalizes valid paraphrases. Cannot detect fluent hallucinations. Useless for multimodal answers (image descriptions have no canonical text). |
| **Human evaluation** | Gold standard for quality. | Does not scale. Cannot run 51 queries after every prompt change. Introduces evaluator bias and inconsistency across sessions. |
| **Embedding similarity to ground truth** | Captures semantic similarity. Fast. | Cannot distinguish a correct answer from a plausible-sounding hallucination if both are semantically close to the ground truth. |

**Why this choice:** The four dimensions measure exactly what matters for a RAG system: Did you get the right answer? Did you make things up? Did you stick to the sources? Did you retrieve the right chunks? BLEU/ROUGE cannot answer any of these questions. The structured rubric (step-by-step reasoning, JSON output with scores and rationale) constrains the judge enough to be useful across runs.

---

### 7.11 No LangChain / LlamaIndex Orchestration

**Decision:** Build the entire retrieval, fusion, reranking, and generation pipeline from scratch. LangChain is allowed ONLY for `RecursiveCharacterTextSplitter` (and in practice, not even used — the project uses a custom chunker).

| | Pros | Cons |
|---|------|------|
| **Hand-built pipeline** | Full control over every parameter (RRF k, rerank threshold, HyDE prompt, chunk overlap). Debuggable — every intermediate result is inspectable. No hidden abstractions or surprise behaviors. Demonstrates genuine understanding of each pipeline stage. Lighter dependency tree. | More code to write and maintain. No community-contributed integrations (e.g., LangChain's 200+ document loaders). Must implement retry logic, batching, and error classification manually. |
| **LangChain** | Fast prototyping. Large ecosystem of integrations. | Opaque abstractions. Difficult to debug when retrieval quality drops. Version churn (breaking changes across releases). Heavy dependency tree. Hides the very engineering decisions this project is meant to demonstrate. |
| **LlamaIndex** | Purpose-built for RAG. Good abstractions for indexing. | Same opacity issues as LangChain. Vendor-neutral abstractions add layers between you and the actual API calls. |

**Why this choice:** This is a portfolio project. The purpose is to demonstrate that the engineer understands hybrid retrieval, RRF fusion, HyDE expansion, and LLM reranking — not that they can call `LangChain.RetrievalQA.from_chain_type()`. Building from scratch also eliminates the largest class of production RAG bugs: unexpected behavior inside framework abstractions.

---

### 7.12 PDF Parsing: PyMuPDF over pdfplumber / PyPDF

**Decision:** Use PyMuPDF (fitz) for all PDF text extraction and TOC reading.

| | Pros | Cons |
|---|------|------|
| **PyMuPDF (fitz)** | Fastest Python PDF library (C-backed). Preserves layout via block-level extraction. Exposes document TOC for section-boundary detection. Handles most PDF variants reliably. | Does not handle scanned PDFs without OCR (would need Tesseract). Block extraction can duplicate text in complex layouts (headers/footers). |
| **pdfplumber** | Better table extraction. Fine-grained character positioning. | Significantly slower (pure Python). No TOC access. Overkill for text extraction. |
| **PyPDF** | Pure Python, no C dependencies. | Slowest of the three. Poor layout preservation. No block-level extraction. |

**Why this choice:** Speed and TOC access. PyMuPDF's block-level extraction plus TOC metadata enables the section-aware chunking strategy. The corpus is text-heavy technical PDFs, not scanned documents, so OCR support is not needed.

---

## 8. Technology Stack

### Backend

| Component | Technology | Version |
|-----------|-----------|---------|
| API Framework | FastAPI + Uvicorn | Latest |
| Embedding | Gemini Embedding 2 (`gemini-embedding-2-preview`) via `google-genai` | Preview |
| Vector Store | ChromaDB | Latest |
| Keyword Search | BM25Okapi (`rank-bm25`) + NLTK Porter Stemmer | Latest |
| PDF Parsing | PyMuPDF (fitz) | Latest |
| Image Processing | Pillow | Latest |
| Video Processing | PySceneDetect + ffmpeg-python + OpenCV | Latest |
| Media Storage | Google Cloud Storage (`google-cloud-storage`) | Latest |
| Token Counting | tiktoken (cl100k_base) | Latest |
| Logging | Loguru | Latest |
| Config | python-dotenv | Latest |
| Telemetry DB | SQLite (via stdlib `sqlite3`) | Built-in |

### Frontend

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | React | 19.x |
| Styling | TailwindCSS | 4.x |
| Charts | Recharts | 3.x |
| Build Tool | Vite | 8.x |
| Linting | ESLint | 9.x |

### Gemini Models Used

| Purpose | Model | Why |
|---------|-------|-----|
| Embedding (all modalities) | `gemini-embedding-2-preview` | Only model with native text + image + video embedding |
| Generation | `gemini-2.5-flash` | Low hallucination, native CoT, 20x cheaper than Pro |
| Reranking | `gemini-2.5-flash` | Multimodal reranking in a single call |
| HyDE expansion | `gemini-2.5-flash` | Fast, cheap, adequate for hypothetical answer generation |
| Image captioning + ROI | `gemini-2.5-flash` | Visual understanding for region detection |
| Video summarization | `gemini-3.1-flash-lite-preview` | Lightweight visual + speech transcription |
| Eval judge | `gemini-2.5-pro` | Higher reasoning quality for evaluation scoring |

---

## 9. Data Models and API Surface

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Environment, ChromaDB status, BM25 index size, eval run count |
| `POST` | `/upload` | Ingest file (PDF/image/video) into the pipeline |
| `POST` | `/query` | Answer a natural language question |
| `GET` | `/eval/runs` | List all eval runs (metadata) |
| `GET` | `/eval/latest` | Full results for most recent run |
| `GET` | `/eval/runs/{run_id}` | Full results for a specific run |
| `GET` | `/metrics/timeseries` | Daily latency, token, and cost aggregates |
| `GET` | `/metrics/summary` | All-time summary statistics |

### Core Pydantic Models

```
QueryRequest        { question: str }
QueryResponse       { answer, sources[], chunks_used, model, media_chunks_degraded }
SourceReference     { filename, page, score, snippet }
UploadResponse      { file_id, filename, size_bytes, status, num_chunks }
EvalRunMeta         { run_id, timestamp, dataset_version, num_queries, avg_latency_ms, ... }
EvalRunDetail       { run_id, timestamp, dataset_version, results[], summary }
TimeseriesPoint     { date, avg_latency_ms, p50, p95, total_queries, tokens, cost }
MetricsSummary      { total_queries, latency percentiles, token averages, cost totals }
```

### Chunk Schema (Internal)

Every chunk flowing through the pipeline carries:

```python
{
    "type": "document" | "image" | "video_clip" | "video_summary",
    "text": str,              # Text content (or caption for images, summary for video)
    "source": str,            # Original filename
    "page": int,              # Page number (PDFs) or segment index
    "chunk_index": int,       # Position within the file
    "modality": str,          # "pdf" | "image" | "video"
    "section_heading": str,   # TOC-derived section (PDFs)
    "document_title": str,    # Document title (PDFs)
    "file_hash": str,         # SHA-256 of the original file
    "gcs_uri": str | None,    # GCS URI for binary media
    "embedding": list[float], # Added by embed_chunks()
}
```

---

## 10. Evaluation Framework

### Golden Dataset

- **51 queries** across 5 categories: factual, multi-hop, cross-modal, adversarial, out-of-scope
- **3 difficulty levels:** easy, medium, hard
- **3 source modalities:** PDF, image, video
- **Corpus:** Transformer technical report (PDF), architecture diagram (PNG), explanatory video (MP4)
- Every ground truth is **verified against actual corpus content** (not generated)

### Judge Dimensions

| Dimension | Scale | What It Measures |
|-----------|-------|------------------|
| Correctness | 0-5 (fractional) | Factual agreement with ground truth |
| Hallucination Rate | 0.0-1.0 | Fraction of claims unsupported by retrieved chunks |
| Faithfulness | 0-5 | How much the answer derives from context vs. parametric knowledge |
| Context Precision | 0.0-1.0 | Fraction of retrieved chunks actually relevant to the query |

### Eval Execution

- Concurrent RAG execution with `asyncio.Semaphore(2)` to respect Gemini rate limits
- Judge scoring runs outside the semaphore (parallelized independently)
- Per-run output: timestamped JSON + SQLite metadata row
- CLI supports `--dry-run` (5 queries) and `--category` filtering

---

## 11. Known Limitations and Future Work

| Limitation | Impact | Upgrade Path |
|------------|--------|-------------|
| ChromaDB is single-machine | Cannot scale past ~1M chunks | Migrate to pgvector with HNSW or managed Pinecone |
| BM25 index lives entirely in memory | Memory pressure at large corpus size | Move to Elasticsearch or OpenSearch |
| No streaming | Complex queries take 10-15s before any response | Add SSE or WebSocket streaming |
| Single-tenant | No document-level access control | Per-user collection namespacing or metadata-filtered search |
| No eval feedback loop | Eval scores are displayed but don't auto-tune the pipeline | Online eval signals → retrieval weight tuning |
| Gemini rate limits under heavy upload | Embedding API throttling | Add job queue (Cloud Tasks / Celery) with backpressure |
| 120s video chunk ceiling | Hard Gemini API constraint; can split coherent scenes | Accept as constraint; mitigate with overlap |
| No Docker | Requires manual local setup | Add Dockerfile + docker-compose for reproducible deployment |

---

## 12. Success Criteria

| Criterion | Target | How Measured |
|-----------|--------|-------------|
| Multimodal ingestion | All 3 modalities (PDF, image, video) ingest without error | Upload each type and verify chunk counts |
| Hybrid retrieval superiority | Hybrid BM25+Vector beats vector-only on exact-match queries | Compare eval scores with BM25 disabled |
| Answer grounding | Every claim in the answer has an inline `[N]` citation | Eval: faithfulness score ≥ 3.5 average |
| Low hallucination | Answers do not fabricate facts | Eval: hallucination rate ≤ 0.15 average |
| Factual correctness | Answers match ground truth | Eval: correctness score ≥ 3.5 average |
| Retrieval precision | Top 5 reranked chunks are relevant | Eval: context precision ≥ 0.6 average |
| Cost efficiency | Per-query cost stays under $0.01 | Metrics dashboard cost tracking |
| Latency | p95 query latency under 15 seconds | Metrics dashboard latency tracking |
| Code quality | No framework orchestration; all pipeline stages hand-built | Code review |
