# Multimodal RAG Notebook

A production-grade Retrieval-Augmented Generation system that lets users upload PDFs,
images, audio, and video, then ask natural language questions about them. The system
chunks and embeds content using Gemini Embedding 2 (natively multimodal), stores vectors
in ChromaDB, and generates grounded answers with Gemini 2.5 Flash. By Week 3 it will
add hybrid BM25 + vector search, LLM-reranking, and a live eval dashboard that scores
answer quality against a 50-query golden dataset using an LLM-as-judge pipeline.

---

## Architecture

```
UPLOAD PIPELINE
───────────────
File (PDF/image/video/audio)
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  Chunk                                                  │
│  PDF  → 800-token chunks (100-token overlap, PyMuPDF)   │
│  Image → whole file                                     │
│  Video → 128 s scene segments                          │
│  Audio → speaker-aligned semantic segments              │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
         Gemini Embedding 2
         (text / image / video / audio)
                   │
                   ▼
              ChromaDB
         (cosine similarity, local)


QUERY PIPELINE
──────────────
User question
  │
  ▼
Gemini Embedding 2  (embed question text)
  │
  ▼
ChromaDB vector search  (top 20)
  │
  ▼
Gemini 2.5 Flash  (generate answer with retrieved context)
  │
  ▼
Answer + source citations
```

---

## Setup

### Prerequisites

- Python 3.9+
- Node.js 18+
- A Gemini API key ([aistudio.google.com](https://aistudio.google.com))
- A Google Cloud Storage bucket

### 1. Clone

```bash
git clone https://github.com/your-username/multimodal-notebook.git
cd multimodal-notebook
```

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```
GEMINI_API_KEY=your_key_here
GCS_BUCKET_NAME=your_bucket_name
```

All other variables have sensible defaults (see `backend/config.py`).

### 4. Frontend

```bash
cd frontend
npm install
```

### 5. Run

**Backend** (from project root):

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
# Listening on http://127.0.0.1:8000
```

**Frontend** (in a second terminal):

```bash
cd frontend
npm run dev
# Listening on http://localhost:5173
```

---

## Example requests

**Upload a file:**

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/document.pdf"
```

**Ask a question:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings in the document?"}'
```

Response shape:

```json
{
  "answer": "...",
  "sources": ["document.pdf#chunk_3", "document.pdf#chunk_7"],
  "chunks_used": 5,
  "model": "gemini-2.5-flash"
}
```

---

## Current Status

| Week | Feature | Status |
|------|---------|--------|
| Week 1 | Basic RAG pipeline (upload → chunk → embed → query) | ✅ Done |
| Week 2 | Hybrid search (BM25 + vector) + LLM reranking | 🔜 Next |
| Week 3 | Eval dashboard + golden dataset + LLM-as-judge | 🔜 Planned |
