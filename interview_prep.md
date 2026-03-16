# Interview Prep — Technical Q&A

Eight questions most likely to come up in a technical screen for this project, with answers that are concise but deep enough to demonstrate genuine understanding.

---

## Q1: Why use hybrid search (BM25 + vector) instead of vector search alone?

**Short answer:** Because vector search and keyword search fail in complementary ways, and the cost of running both is low.

Vector search is excellent at semantic similarity — it understands that "monetary policy tightening" and "interest rate hikes" are related. But it tends to smooth over exact terms. A query for "GPT-4o" or "v2.3.1" or a specific person's name may not surface the right chunk because the embedding collapses those tokens toward their conceptual neighbors. BM25 uses TF-IDF-style scoring on exact tokens, so it excels precisely where vector search is weakest: proper nouns, codes, acronyms, and rare technical terms.

In IR benchmarks, hybrid consistently outperforms either method alone on heterogeneous document corpora — which is exactly what this system handles. The overhead is minimal: BM25 is an in-memory index built at ingestion time, query latency is sub-millisecond, and both searches run in parallel before fusion.

---

## Q2: How does Reciprocal Rank Fusion work, and why not just take a weighted average of scores?

**Short answer:** RRF uses ranks, not scores, because scores from different retrieval systems aren't on comparable scales.

The formula is: `rrf_score(doc) = Σ 1 / (k + rank_i)` across all lists the document appears in. `k=60` is a smoothing constant from the original Cormack et al. 2009 paper — it prevents a rank-1 document from completely dominating, and was empirically found to work well across diverse retrieval tasks.

Weighted score averaging fails because BM25 scores are unbounded (depend on corpus statistics and document length) while cosine similarity scores are bounded [0, 1]. Normalizing each system's scores before combining is possible but fragile — it requires tuning and breaks when score distributions shift as documents are added. Ranks are always comparable: rank 1 is the best result from that system regardless of how confident it was.

RRF also naturally handles documents that appear in only one list, giving them partial credit proportional to their rank in that list.

---

## Q3: What is HyDE and why does it help?

**Short answer:** HyDE solves the question/answer space mismatch in embedding retrieval.

When a user asks "What caused the 2008 financial crisis?", that question lives in question space semantically — it's short, interrogative, and contains no actual claims. The chunks in the index are in answer space — they contain statements like "The crisis was precipitated by..." Embedding a question and comparing it to answer-space chunks produces suboptimal similarity scores.

HyDE has the LLM generate a plausible 2–3 sentence answer to the question, then embeds *that* answer and uses it as the query vector. The hypothetical answer lands in answer space and is geometrically closer to the real answer chunks.

The risk is that the LLM generates a confidently wrong hypothetical, which could retrieve off-topic chunks. In practice this is rare because: (1) the hypothetical only needs to be in the right semantic neighborhood, not factually correct, and (2) BM25 still runs against the original query, so RRF fusion corrects for a bad hypothetical by surfacing the keyword-matched chunks.

---

## Q4: Why does the reranker outperform just keeping the top-5 from RRF?

**Short answer:** Retrieval optimizes for recall (find anything relevant); reranking optimizes for precision (rank the most relevant first).

BM25 and vector search are fast approximate methods — they use precomputed indexes and can't afford to read the full content of every candidate at query time. The reranker operates on a much smaller candidate set (20 chunks) and has the luxury of reading all of them simultaneously, comparing them against each other rather than scoring each in isolation.

More importantly, this system's reranker is multimodal. When chunks include images or video frames, the reranker sends the actual bytes to Gemini and can reason about visual content directly. A text-similarity-based retrieval step can't do that — it can only compare text embeddings of image captions.

The computational cost is justified: one LLM API call on 20 short chunks is fast (~1s), and the quality improvement from sending 5 reranked chunks instead of 5 random RRF chunks to the generator is significant. Garbage in the context directly produces hallucinations out.

---

## Q5: How do you handle multimodal content — images and video — in a text-centric retrieval pipeline?

**Short answer:** Gemini Embedding 2 supports all modalities natively, so the pipeline treats image and video chunks the same as text chunks at the embedding layer.

At ingestion, images produce two chunks: a global chunk (full image + generated caption) and local chunks per detected region of interest. The image bytes and a text caption are both embedded by Gemini Embedding 2 as a single multimodal embedding — not as separate text-then-combine. Video clips are embedded as native MP4 bytes (up to 128 seconds).

At retrieval, BM25 only sees text, but vector search retrieves image and video chunks using their multimodal embeddings. The reranker receives the actual bytes, not just text summaries. The generator also receives inline bytes for any image or video chunk in the top 5, so it can reason about visual content in its final answer.

The fallback chain when GCS fetch fails: text summary → caption → skip with a degraded count. The `media_chunks_degraded` field in the response makes this visible to the caller.

---

## Q6: How would you scale this to production with millions of documents?

**Short answer:** Replace local components with managed services, add async job processing, and introduce caching at the right layers.

**Vector store:** ChromaDB is local-process, single-machine. Replace with a managed HNSW-backed service (Pinecone, Weaviate) or pgvector with an HNSW index. At tens of millions of vectors, you also need to shard by collection or metadata.

**BM25 / keyword search:** The in-memory BM25 index can't fit on one machine at scale. Replace with Elasticsearch or OpenSearch, which support distributed inverted indexes with the same BM25 scoring.

**Ingestion:** Replace synchronous upload with a job queue (Cloud Tasks, Pub/Sub, Celery). The upload endpoint enqueues a job and returns immediately; workers process ingestion asynchronously. This decouples latency from file size and handles bursts without blocking the API.

**Embedding throughput:** Gemini Embedding API has quota limits. At scale, batch workers with per-key rate limiting and exponential backoff. Consider a dedicated embedding cache (Redis, keyed by chunk SHA-256) to avoid re-embedding duplicate content across users.

**Query latency:** Add a semantic cache for repeated or similar queries. For popular queries, the BM25 + vector search results can be cached for minutes.

**Multi-tenancy:** Namespace ChromaDB collections by user or team. Add metadata filtering so a user can only retrieve from their own documents.

---

## Q7: How do you prevent hallucinations, and how would you measure them?

**Short answer:** Constrain generation by instruction, validate citations programmatically, and measure with LLM-as-judge.

**Prevention:** The system prompt explicitly instructs the model to answer only from provided sources, cite every claim with `[N]`, and return a specific refusal phrase if the information isn't in the context. The model uses a CoT thinking budget that forces explicit reasoning steps before producing the final answer — this makes it harder to confabulate because the model has to "show its work."

**Detection:** After generation, citation indices are extracted from the response and validated against the actual chunks sent to the model. Any `[N]` outside the valid range `[1..5]` is a hallucinated citation and is logged as a warning with the full response. This catches the most common hallucination pattern: citing a source that was never in the context.

**Measurement:** The eval dashboard runs a 50-query golden dataset and uses Gemini 2.5 Pro as judge, scoring each response on five dimensions: correctness (does the answer match ground truth?), faithfulness (are all claims grounded in the context?), hallucination rate (does the answer assert facts not in the sources?), context precision (are the retrieved chunks actually relevant?), and latency. This gives a quantitative baseline to compare prompt changes, retrieval configurations, or model upgrades.

---

## Q8: Why Gemini 2.5 Flash for generation instead of a more powerful model?

**Short answer:** It has lower hallucination rates than 1.5 Pro, native multimodal support, and is 20× cheaper — cost efficiency matters at eval-time when you're running 50 queries in batch.

The key insight is that generation quality in a RAG system is bounded by retrieval quality. If you send the right 5 chunks to the model, even a smaller model produces an excellent answer — the heavy lifting has already been done. Upgrading to a larger model helps at the margin but doesn't fix bad retrieval. Investing in retrieval (HyDE, reranking, hybrid search) has higher ROI than upgrading the generator.

Gemini 2.5 Flash supports a thinking budget (extended CoT), which gives it reasoning capability typically associated with larger models at a fraction of the cost. It also handles image and video bytes natively, which is essential for this system's multimodal reranking and generation steps — a text-only model would require a preprocessing step to convert all media to text before generation.

For the eval pipeline specifically, running a 50-query batch against a model 20× cheaper means you can iterate on prompts and retrieval configurations much more aggressively without budget constraints.
