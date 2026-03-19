# Multimodal RAG Notebook

A production-grade Retrieval-Augmented Generation system that lets you upload PDFs, images, videos, and audio files and ask natural language questions about them. It uses a hybrid BM25 + vector search pipeline with LLM reranking and grounded answer generation via Gemini 2.5 Flash.

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║  INGESTION PIPELINE                                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  File Upload (PDF / PNG / JPEG / MP4 / MOV / MP3 / WAV / M4A)       ║
║       │                                                              ║
║       ▼                                                              ║
║  Chunking                                                            ║
║  ├── PDF      → 800-token chunks (100 overlap, TOC-aware)            ║
║  ├── Image    → global chunk + per-region crops (Gemini Flash ROI)   ║
║  ├── Video    → 120s scenes → video clip chunk + text summary chunk  ║
║  └── Audio    → semantic-boundary segments → MP3 clips + transcript  ║
║       │                                                              ║
║       ▼                                                              ║
║  Binary media (images, video frames, audio) → Google Cloud Storage  ║
║       │                                                              ║
║       ▼                                                              ║
║  Gemini Embedding 2  (text: 20/batch · images: 6/batch · video: 1)  ║
║       │                                                              ║
║       ├──▶ ChromaDB  (cosine similarity, persistent)                 ║
║       └──▶ BM25 Index  (Porter stemming, persisted to disk)          ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  QUERY PIPELINE                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  User Question                                                       ║
║       │                                                              ║
║       ▼                                                              ║
║  HyDE — Gemini Flash writes a 2–3 sentence hypothetical answer       ║
║  → embed the hypothetical answer (not the raw question)              ║
║       │                                                              ║
║       ├──▶ BM25 keyword search      → top 20 results                 ║
║       └──▶ Vector search (ChromaDB) → top 20 results                 ║
║                │                                                     ║
║                ▼                                                     ║
║  Reciprocal Rank Fusion (RRF, k=60)                                  ║
║  → merged ranked list of top 20 unique chunks                        ║
║       │                                                              ║
║       ▼                                                              ║
║  Reranking — single multimodal Gemini Flash call                     ║
║  Scores all 20 candidates simultaneously → keeps top 5               ║
║       │                                                              ║
║       ▼                                                              ║
║  Gemini 2.5 Flash — Chain-of-Thought generation                      ║
║  Strict grounding: cite every claim as [N], refuse to hallucinate    ║
║       │                                                              ║
║       ▼                                                              ║
║  Answer + source citations + latency + token metrics                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| API framework | FastAPI + Uvicorn | Async-native, automatic OpenAPI docs, minimal boilerplate |
| Embedding model | Gemini Embedding 2 (`gemini-embedding-2-preview`) | Only embedding model with native text + image + video support in a single API |
| Vector store | ChromaDB (cosine similarity) | Local-first, no infra overhead, persistent — appropriate for portfolio scale |
| Keyword search | BM25Okapi + Porter stemming | Exact-match retrieval catches acronyms, names, and codes that vector search misses |
| Rank fusion | Reciprocal Rank Fusion (RRF) | Parameter-free; proven to beat weighted score combination in IR benchmarks |
| Reranker | Gemini 2.5 Flash | Multimodal — can score image and video chunks directly, not just text |
| Generator | Gemini 2.5 Flash | ~3.3% hallucination rate, native multimodal, 20× cheaper than 1.5 Pro |
| Reasoning | Chain-of-Thought thinking (1024 token budget) | Forces model to reason before answering; reduces unsupported claims |
| Media storage | Google Cloud Storage | Decouples binary blobs from the vector store; enables rehydration at query time |
| PDF parsing | PyMuPDF (fitz) | Fastest Python PDF library; preserves layout and TOC structure |
| Token counting | tiktoken (cl100k_base) | Consistent chunk boundary enforcement |
| Scene detection | PySceneDetect | Reliable scene boundary detection for video splitting |
| Frontend | React 18 + TailwindCSS + Vite | Fast iteration; Vite HMR; Tailwind avoids CSS drift |

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- A Google AI Studio API key (Gemini)
- A Google Cloud project with a GCS bucket (for media storage)
- `ffmpeg` on `PATH` (for video frame extraction)

### Environment Variables

Create `.env` in the **project root** (not inside `backend/`):

```bash
# Required
GEMINI_API_KEY=your_google_ai_studio_key_here
GCS_BUCKET_NAME=your_gcs_bucket_name_here

# Optional (defaults shown)
ENVIRONMENT=development
HOST=127.0.0.1
PORT=8000
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=rag_chunks
TMP_UPLOAD_DIR=./tmp
BM25_TOP_K=20
VECTOR_TOP_K=20
RERANK_TOP_K=5
EVAL_DB_PATH=./backend/eval/eval_results.db
EVAL_DATASET_PATH=./backend/eval/golden_dataset.json
EVAL_RESULTS_DIR=./evals
EVAL_JUDGE_MODEL=gemini-2.5-pro
GENERATION_THINKING_BUDGET=1024
```

### Install Dependencies

```bash
# Backend
cd backend
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# NLTK data (Porter stemmer)
python -c "import nltk; nltk.download('punkt')"

# Frontend
cd ../frontend
npm install
```

### Run Locally

```bash
# Terminal 1 — backend
cd backend
source .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev
```

Open `http://localhost:5173`. API at `http://localhost:8000`.

Health check: `curl http://localhost:8000/health`

### Running the Eval Suite

```bash
# Full run — all 50 golden queries
python -m backend.eval.runner

# Dry run — first 5 queries only (fast sanity check, ~2 min)
python -m backend.eval.runner --dry-run

# Filter by category
python -m backend.eval.runner --category factual
python -m backend.eval.runner --category multi-hop
python -m backend.eval.runner --category cross-modal
python -m backend.eval.runner --category adversarial
python -m backend.eval.runner --category out-of-scope
```

Results are written to `./evals/results_<timestamp>.json`. Run metadata (latency percentiles, judge scores) is persisted to `./backend/eval/eval_results.db` and surfaced live in the **Eval Dashboard** tab of the UI.

### Quick API Test

```bash
# Upload a PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?"}'
```

Response shape:

```json
{
  "answer": "The main findings show... [1][2]",
  "sources": [
    {"filename": "report.pdf", "page": 4, "score": 0.91, "snippet": "..."},
    {"filename": "report.pdf", "page": 7, "score": 0.87, "snippet": "..."}
  ],
  "chunks_used": 5,
  "model": "gemini-2.5-flash",
  "media_chunks_degraded": 0
}
```

---

## How It Works

### Stage 1 — Ingestion and Chunking

When you upload a file, the system breaks it into retrievable chunks appropriate for the modality.

**PDFs** are split into 800-token windows with 100-token overlap using PyMuPDF. The chunker reads the document's table of contents and resets boundaries at section headings so a chunk never straddles two unrelated topics. Mega-paragraphs over 800 tokens are hard-split.

**Images** produce two types of chunks: one global chunk containing the full image plus a Gemini-generated caption (visual description + any transcribed text), and one local chunk per region of interest detected by Gemini Flash. This dual-stream approach means a question about a specific chart can match the precise crop rather than the whole-page embedding.

**Videos** are split at scene boundaries (PySceneDetect) with a 120-second hard ceiling imposed by the Gemini Embedding API. Each scene produces two chunks: a native MP4 clip embedded directly as video (Gemini understands motion and audio together), and a text summary chunk so BM25 can also retrieve it by keyword.

**Audio** is transcribed by Gemini Flash with word-level timestamps and speaker labels, then split at semantic boundaries detected by comparing sentence embeddings. Short same-speaker segments under 60 seconds are merged, with a 75-second hard ceiling. Each segment is stored as an MP3 in GCS alongside its transcript.

All binary media is uploaded to GCS. The URI is stored with the chunk for rehydration at query time.

After chunking, chunks are embedded via Gemini Embedding 2 in batches (20 text, 6 images, 1 video/audio per request). Embeddings go into ChromaDB. Text chunks are simultaneously added to a BM25 index persisted to disk and loaded at startup.

### Stage 2 — HyDE Query Expansion

When you ask a question, the system doesn't immediately embed your question and search. It uses **Hypothetical Document Embedding (HyDE)**: Gemini Flash writes a 2–3 sentence hypothetical answer to your question. That answer is then embedded and used as the search vector.

The intuition: your question lives in question space, but the indexed chunks live in answer space. Embedding a hypothetical answer bridges that gap — "The crash was caused by over-leveraged mortgage securities" is geometrically closer to the real answer chunks than the bare question "What caused the crash?" If HyDE fails for any reason, the system falls back to embedding the raw question.

### Stage 3 — Hybrid Search (BM25 + Vector)

Two independent searches run against the full corpus and each return their top 20 results:

**BM25** is a classical keyword scoring algorithm (TF-IDF family, Okapi variant). It rewards chunks containing the exact words from your query at high frequency, weighted by how rare those words are across the corpus. It's essential for queries involving proper names, product codes, version numbers, or acronyms — terms a neural embedding might generalize away.

**Vector search** finds chunks semantically similar to the HyDE-expanded query embedding, even without shared vocabulary. It handles synonyms, paraphrases, and conceptual matches that keyword search would miss entirely.

### Stage 4 — Reciprocal Rank Fusion (RRF)

The two ranked lists are merged using **Reciprocal Rank Fusion**: each chunk earns `1 / (k + rank)` from each list it appears in, where `k=60` is the smoothing constant from Cormack et al. 2009. Scores are summed across lists. A chunk ranking #3 in BM25 and #5 in vector scores higher than a chunk appearing in only one list at rank #1.

RRF requires no learned weights. It consistently outperforms weighted score combination because raw similarity scores from BM25 and a neural embedding model aren't on comparable scales — but ranks are.

### Stage 5 — LLM Reranking

The top 20 RRF candidates go to a **reranker**: a single Gemini Flash call that receives all 20 chunks simultaneously and scores each one for relevance to the original query. Unlike retrieval, the reranker sees all candidates at once and can make relative comparisons rather than scoring each in isolation.

The reranker is multimodal. Image chunks are sent as actual JPEG bytes, video chunks include their first frame, audio chunks include their transcript. The model can reason about whether a diagram directly answers the question rather than relying on text overlap.

Output: top 5 chunks with `rerank_score` (0.0–1.0). These 5 are the only chunks the generation model sees — passing all 20 would dilute context quality and inflate token costs.

### Stage 6 — Grounded Answer Generation

The 5 reranked chunks are assembled into a multimodal context and sent to **Gemini 2.5 Flash** with a strict system prompt: answer only from the provided sources, cite every claim inline as `[N]`, and return "I don't have enough information" if the answer isn't in the context.

The model uses **Chain-of-Thought thinking** (1024 token budget) before producing its final response. This internal reasoning step reduces hallucinations and improves faithfulness. The response includes the answer text, source citations with page numbers and relevance scores, and operational metrics (chunks used, model name, degraded media count).

---

## Key Engineering Decisions

### Hybrid Search over Vector-Only

Pure vector search excels at semantic similarity but fails on exact-match queries — model names, paper titles, version numbers, and acronyms are averaged away during embedding and produce poor cosine similarity. BM25 captures these signals with its TF-IDF weighting. Reciprocal Rank Fusion merges both ranked lists without requiring score normalisation, which is critical because raw BM25 and cosine scores live on incompatible scales while ranks are always comparable.

### HyDE (Hypothetical Document Embeddings)

A raw user query ("what causes gradient vanishing?") and its ideal answer chunk ("gradient vanishing occurs when…") occupy different regions of embedding space — questions and answers have different linguistic form. HyDE closes this gap by generating a plausible hypothetical answer first and embedding that instead of the raw query. In practice this shifts retrieval recall by roughly 10–20 % on long-form factual questions at no additional latency on the critical path (HyDE runs concurrently with BM25 query tokenisation).

### LLM-as-Judge over BLEU / ROUGE

BLEU and ROUGE measure n-gram overlap between generated and reference text. They penalise valid paraphrases and, critically, cannot detect factual hallucinations that are fluent and lexically similar to the ground truth. An LLM judge evaluates semantic correctness, groundedness in the retrieved context, and whether claims in the answer are supported by the cited source — the three properties that matter for a RAG system. The tradeoff is cost and non-determinism; both are acceptable given the 50-query evaluation cadence and the structured 0–5 scoring rubric that constrains judge variance.

### Gemini Embedding 2 over OpenAI Embeddings

The corpus contains PDFs, images, and video. Maintaining separate embedding models per modality would require separate ChromaDB namespaces and modality-routing logic at query time. Gemini Embedding 2 accepts text, image bytes, and video clips natively and produces embeddings in a shared vector space. This makes cross-modal retrieval (a diagram answering a text query) possible with a single ChromaDB collection and no routing code.

---

## Interview Talking Points

- **End-to-end pipeline, no framework magic.** The hybrid search, RRF fusion, HyDE expansion, and LLM reranking are all implemented from scratch rather than delegated to LangChain or LlamaIndex. This demonstrates understanding of the trade-offs at each stage and makes the system easier to debug and iterate on.

- **Multimodal-native design.** Using Gemini Embedding 2 as the single embedding model across text, images, and video is an architectural choice that enables cross-modal retrieval — a diagram in a PDF answering a text question — with one vector index and no modality-routing logic. This is a non-trivial design decision with real implications for index structure and query latency.

- **Production eval pipeline.** The LLM-as-judge setup with four independent metrics (correctness, hallucination rate, faithfulness, context precision) mirrors what a production ML team would ship. It makes regressions detectable and prompt changes measurable — both essential for responsible iteration on a RAG system at any scale.

- **Cost-aware architecture.** Every model choice was made with cost and latency in mind. Gemini 2.5 Flash is ~20× cheaper than 1.5 Pro with comparable faithfulness. The reranker reuses the same Flash model as generation, avoiding an additional API dependency. Chunking aggressively deduplicated via SHA-256 prevents re-embedding unchanged files.

- **Retrieval quality fundamentals.** The BM25 + vector + RRF + reranker stack is the same pattern used in production search systems at scale. Being able to explain why each layer exists and what failure mode it addresses — exact-match gaps, semantic drift, score scale mismatch, context window budget pressure — demonstrates depth beyond "I used a RAG library."

---

## Known Limitations

**Scale.** ChromaDB is a local vector store. At millions of chunks, memory footprint and query latency become problematic. Production would use a managed vector DB (Pinecone, Weaviate, pgvector with HNSW indexing).

**BM25 lives in memory.** The pickled BM25 index is loaded in full at startup. At scale this moves to a dedicated search service (Elasticsearch, OpenSearch).

**No streaming.** Answers are returned only when generation is complete. With the CoT thinking budget, complex queries can take 10–15 seconds. Production would stream tokens via SSE or WebSockets.

**Single-tenant.** One shared ChromaDB collection, no document-level access control. Multi-tenancy requires per-user collection namespacing or metadata-filtered search.

**No eval feedback loop.** The eval dashboard scores response quality but doesn't update retrieval weights or prompts automatically. A production system would use online eval signals to tune the pipeline.

**Rate limits.** Heavy upload volumes will hit Gemini Embedding API quotas. Production would add a job queue (Cloud Tasks, Celery) with backpressure and per-user rate limiting.

**Video segment ceiling.** The 120-second maximum per video chunk is a hard Gemini Embedding API constraint. Longer scenes are hard-cut, which can split coherent dialogue mid-sentence.
