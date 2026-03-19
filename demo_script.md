# Demo Script — 3-Minute Portfolio Walkthrough

**Total runtime:** ~3 minutes
**Format:** Four timed beats. Each beat has talking points and what to have visible on screen.

**Before you start:**
- Backend running (`uvicorn backend.main:app --reload`), terminal visible side-by-side with browser
- Browser at `http://localhost:5173`, starting on the **Chat** tab
- A technical PDF ready to drag-and-drop — ideally 10–20 pages with a TOC, named entities (product names, version numbers), and conceptual prose (e.g., an earnings report or research paper)
- Eval Dashboard tab already loaded so the chart renders instantly when you switch

---

## Beat 1 — What the system does and why (0:00 – 0:30)

**Talking points:**

> "This is a multimodal RAG system — Retrieval-Augmented Generation. You upload a document, image, or video. The system indexes it using both keyword search and semantic vector search. When you ask a question, it retrieves the most relevant content from your corpus, reranks it, and sends the top five chunks to Gemini 2.5 Flash to generate a grounded answer with inline citations."

> "The key word is *grounded*. The model is constrained to cite every claim as [1], [2], etc. and is instructed to say 'I don't have enough information' rather than guess. That constraint is what separates a useful RAG system from a hallucination machine. The second tab is an eval dashboard that quantifies how well that constraint holds up — using a 50-query golden dataset scored by Gemini-as-judge on correctness, faithfulness, and hallucination rate."

---

## Beat 2 — Live demo: upload → retrieve → answer (0:30 – 1:30)

### Upload (15 sec)

Drag the PDF onto the upload zone. Watch the status badge: **Uploading → Indexing → Ready**.

> "The file hits `/upload`. PyMuPDF parses it TOC-aware — chunk boundaries reset at section headings so a chunk never straddles two unrelated topics. Gemini Embedding 2 embeds the chunks in batches of 20. At the same time, the text chunks go into a BM25 keyword index that's persisted to disk and loaded at startup. ChromaDB is the single source of truth for embeddings."

**Watch in the terminal:**
```
INFO | Chunked report.pdf → 52 chunks
INFO | Embedding batch 1/3 (20 chunks)
INFO | BM25 index rebuilt — 52 documents total
INFO | Ingestion complete | elapsed=9.1s
```

### Ask a keyword-dependent question (20 sec)

Type a query with a specific named entity: *"What were the Q3 revenue figures for the Enterprise segment?"*

> "This query is designed to stress-test keyword retrieval — 'Q3 revenue' and 'Enterprise segment' appear verbatim in the document. A pure vector search might surface conceptually similar content and miss the exact table row. BM25 catches it."

> "But before searching, the system runs HyDE — Hypothetical Document Embedding. Gemini Flash writes a short hypothetical answer and we embed *that* instead of the raw question. A question and its answer occupy different regions of embedding space; HyDE bridges that gap."

**Show the logs:**
```
INFO | HyDE | hypothesis="Enterprise segment revenue for Q3 reached $2.1B..."
INFO | BM25 top-3:   [chunk_12 score=4.21, chunk_8 score=3.77, chunk_19 score=2.88]
INFO | Vector top-3: [chunk_12 score=0.91, chunk_31 score=0.84, chunk_8 score=0.80]
INFO | RRF fusion → 20 unique candidates
INFO | Rerank complete | top5=[12, 8, 19, 31, 7] | elapsed=1.2s
```

> "chunk_12 ranks #1 in both BM25 and vector — that chunk gets a heavily boosted RRF score. The reranker then sees all 20 candidates at once and confirms it belongs at the top."

### Show the answer and citations (25 sec)

Point to the answer pane.

> "The answer cites [1] and [2] — those map to chunk_12 on page 4 and chunk_8 on page 7. Click either citation to see the exact excerpt that supports the claim. If you ask something the document doesn't cover, the model returns 'I don't have enough information' — I can show that too if you want."

---

## Beat 3 — Eval Dashboard walkthrough (1:30 – 2:30)

Switch to the **Eval Dashboard** tab.

### Metrics table (20 sec)

> "Each row is one eval run — the pipeline ran against 50 hand-crafted queries with ground-truth answers. The columns are correctness, faithfulness, hallucination rate, and context precision, each scored 0–5 by Gemini-as-judge. The latest run is at the top."

> "I chose LLM-as-judge over BLEU or ROUGE because BLEU measures n-gram overlap — it penalises valid paraphrases and can't detect fluent hallucinations. An LLM judge evaluates semantic correctness and whether claims are grounded in the retrieved sources, which are the two things that actually matter for a RAG system."

### Before/after comparison (20 sec)

Select two runs with different configurations to show the diff table.

> "This table is the core workflow for prompt engineering. I ran the eval before adding HyDE and after. Correctness went from 3.2 to 3.8 and hallucination rate dropped from 0.9 to 0.4. That's a measurable improvement from one retrieval change — and now it's in the database so I can always come back to it."

### Latency chart (20 sec)

Point to the p50/p95 latency trend.

> "p50 is around 3.5 seconds, p95 is around 9 seconds. The tail latency is from queries that trigger longer CoT thinking budgets. This chart makes it visible when a model config change blows up latency — for example, increasing the thinking budget from 1024 to 4096 tokens pushes p95 past 15 seconds."

---

## Beat 4 — Closing: what's next (2:30 – 3:00)

> "Three things I'd add to take this to production scale:"

> "**Vertex AI Matching Engine** instead of ChromaDB — managed approximate nearest-neighbour search that handles hundreds of millions of vectors with sub-100ms latency guarantees. ChromaDB is the right call for local development but doesn't scale beyond a few million chunks."

> "**Streaming responses** via Server-Sent Events — right now the answer only appears when generation is complete. With Gemini's streaming API and a small SSE endpoint on the FastAPI side, the first tokens appear in under a second. The UX difference is significant for complex queries with long CoT."

> "**Fine-tuned reranker** — the current reranker is a general-purpose Gemini Flash call. Training a lightweight cross-encoder on domain-specific query/document pairs from the eval logs would improve top-5 precision by 10–15 % based on published numbers for similar setups, at a fraction of the cost per query."
