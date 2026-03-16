# Demo Script — 3-Minute Live Walkthrough

**Goal:** Show the full pipeline end-to-end in three minutes. Upload a document, run three queries that each highlight a different technical capability, and point to the console logs to make the pipeline visible.

**Recommended test document:** A technical PDF with tables and charts (e.g., an earnings report, research paper, or product specification). Ideal: 10–20 pages, has a table of contents, contains specific named entities (product names, version numbers, dollar figures) AND conceptual prose.

---

## Before You Start (30 seconds of prep)

1. Have the terminal visible side-by-side with the browser tab.
2. Run the backend with `--log-level info` so all loguru output is visible.
3. Clear the terminal so logs from the demo are clean.
4. Have the document ready to drag-and-drop.

---

## Step 1 — Upload the Document (45 seconds)

**What to do:** Drag the PDF onto the upload zone. Watch the status label change: "Uploading..." → "Indexing..." → "Ready — N chunks indexed".

**Talking points:**

> "The file hits the `/upload` endpoint. The backend reads the PDF with PyMuPDF, extracts the table of contents, and splits the text into 800-token chunks with 100-token overlap, resetting at section boundaries so no chunk straddles two topics. While that's happening, any images in the document get a dual treatment: one embedding for the full page and one per detected region of interest — so if you ask about a specific chart later, the system can surface that exact crop."

> "After chunking, the backend calls Gemini Embedding 2 in batches of 20 and writes the embeddings to ChromaDB. In parallel, the text chunks are added to a BM25 keyword index that persists to disk. That BM25 index is what lets us do keyword matching later — it's rebuilt incrementally and loaded at startup."

**Watch for in the logs:**
```
INFO  | Chunked <filename> → 47 chunks
INFO  | Embedding batch 1/3 (20 chunks)...
INFO  | BM25 index rebuilt — 47 documents total
INFO  | Ingestion complete | file_hash=abc123 | chunks=47 | elapsed=8.3s
```

---

## Step 2 — Query 1: Keyword-Dependent Question (45 seconds)

**Run this query:** Something with a specific named entity that only appears verbatim in the document — a product name, model number, version string, or proper noun. Example: *"What were the Q3 revenue figures for the Enterprise segment?"*

**Talking points:**

> "This query is designed to stress-test keyword retrieval. The phrase 'Q3 revenue' and 'Enterprise segment' are exact strings that appear in the document. A pure vector search might surface conceptually similar content but miss the specific table row with those figures. BM25 catches it because it rewards exact term frequency weighted by rarity across the corpus."

> "Notice in the logs: the system runs HyDE first — Gemini Flash writes a 2–3 sentence hypothetical answer and we embed *that* instead of the raw question. The BM25 search still runs against the original query text, not the hypothetical. Then we fuse the two ranked lists with Reciprocal Rank Fusion."

**Watch for in the logs:**
```
INFO  | HyDE expansion | hypothesis="Enterprise segment revenue for Q3 reached..."
INFO  | BM25 top-3: [chunk_12 score=4.21, chunk_8 score=3.77, chunk_19 score=2.88]
INFO  | Vector top-3: [chunk_12 score=0.89, chunk_31 score=0.84, chunk_8 score=0.81]
INFO  | RRF fusion → 20 unique candidates
INFO  | Rerank complete | top5=[12, 8, 19, 31, 7] | elapsed=1.2s
```

> "See chunk_12 ranking #1 in both BM25 and vector? When a chunk wins both lists it gets a heavily boosted RRF score. The reranker then confirms it belongs at the top. The final answer cites [1] for that chunk specifically."

---

## Step 3 — Query 2: Semantic / Conceptual Question (45 seconds)

**Run this query:** Something conceptual that doesn't match any exact phrase in the document. Example: *"What are the biggest risks to growth next year?"* or *"How does the company plan to stay competitive?"*

**Talking points:**

> "This is the opposite of the last query. No exact phrase in the document says 'risks to growth.' But the document probably has paragraphs about 'macroeconomic headwinds,' 'supply chain uncertainty,' or 'competitive pressure.' Vector search handles this because Gemini Embedding 2 understands that 'risks to growth' and 'macroeconomic headwinds' are semantically related."

> "HyDE is especially powerful here. The raw question 'What are the biggest risks?' is very short and vague. The hypothetical answer Gemini generates is 2–3 sentences about the specific kinds of risks a company of this type faces — and *that* embedding lands much closer to the relevant chunks in embedding space."

**Watch for in the logs:**
```
INFO  | HyDE expansion | hypothesis="The company faces headwinds including rising input costs,
       competitive pressure from new entrants, and potential regulatory changes..."
INFO  | Rerank complete | top5=[3, 22, 15, 9, 28] | elapsed=1.4s
INFO  | Generation complete | tokens_used=1847 | thinking_tokens=412 | elapsed=6.1s
```

> "The answer should cite 2–3 distinct sections of the document. That's the grounding working — the model is instructed to only claim what the sources support and to say 'I don't have enough information' otherwise."

---

## Step 4 — Query 3: Multi-Hop / Citation Quality Question (30 seconds)

**Run this query:** Something that requires synthesizing information from two different parts of the document. Example: *"How does the company's stated strategy compare to its actual capital allocation?"* or *"Are the growth projections consistent with the risks described elsewhere in the report?"*

**Talking points:**

> "This query tests citation quality. A good RAG system doesn't just retrieve one relevant chunk — it pulls from multiple sections and synthesizes them. The reranker's job here is critical: it sees all 20 RRF candidates at once and can make relative quality judgments. A chunk that partially addresses the question might rank lower than two chunks that together form a complete answer."

> "Watch the source citations in the response. You should see [1] and [2] pointing to different pages — the strategy section and the financial section, for example. The Gemini 2.5 Flash CoT thinking budget means the model reasons through how those two pieces fit together before producing the answer."

**Watch for in the logs:**
```
INFO  | Rerank complete | top5=[4, 17, 6, 22, 11] | elapsed=1.3s
INFO  | Context build | text_chunks=3 | image_chunks=2 | degraded=0
INFO  | Citation validation | valid=[1,2,3,4,5] | hallucinated=[]
INFO  | Generation complete | chunks_used=5 | elapsed=8.7s
```

> "The `hallucinated=[]` line is the citation validator — it checks that every `[N]` the model outputs corresponds to a real chunk we sent. If the model invented a citation, it would appear here and the system would log a warning."

---

## Wrap-Up (15 seconds)

> "The full pipeline is: HyDE expansion → BM25 + vector search in parallel → RRF fusion → LLM reranking → CoT generation with strict grounding. Every stage exists because removing it measurably hurts answer quality: vector-only misses exact matches, no reranking sends noise to the generator, and no grounding constraint produces hallucinations. The eval dashboard — coming in Week 3 — will quantify exactly how much each stage contributes using a 50-query golden dataset scored by Gemini-as-judge."
