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

Create `backend/.env`:

```bash
# Required
GEMINI_API_KEY=your_google_ai_studio_key_here
GCS_BUCKET_NAME=your_gcs_bucket_name_here

# Optional (defaults shown)
ENVIRONMENT=development
HOST=127.0.0.1
PORT=8000
CHROMA_PERSIST_DIR=./chroma_db
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

## Known Limitations

**Scale.** ChromaDB is a local vector store. At millions of chunks, memory footprint and query latency become problematic. Production would use a managed vector DB (Pinecone, Weaviate, pgvector with HNSW indexing).

**BM25 lives in memory.** The pickled BM25 index is loaded in full at startup. At scale this moves to a dedicated search service (Elasticsearch, OpenSearch).

**No streaming.** Answers are returned only when generation is complete. With the CoT thinking budget, complex queries can take 10–15 seconds. Production would stream tokens via SSE or WebSockets.

**Single-tenant.** One shared ChromaDB collection, no document-level access control. Multi-tenancy requires per-user collection namespacing or metadata-filtered search.

**No eval feedback loop.** The eval dashboard scores response quality but doesn't update retrieval weights or prompts automatically. A production system would use online eval signals to tune the pipeline.

**Rate limits.** Heavy upload volumes will hit Gemini Embedding API quotas. Production would add a job queue (Cloud Tasks, Celery) with backpressure and per-user rate limiting.

**Video segment ceiling.** The 120-second maximum per video chunk is a hard Gemini Embedding API constraint. Longer scenes are hard-cut, which can split coherent dialogue mid-sentence.
