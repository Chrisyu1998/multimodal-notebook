"""
HyDE A/B comparison script.

For each of 5 test queries, runs vector search twice:
  - WITH HyDE:    embed the Gemini-generated hypothetical document
  - WITHOUT HyDE: embed the raw query string

All queries are scoped to content that exists in pdftest1.pdf
("Attention Is All You Need", Vaswani et al. 2017) so every comparison
has a knowable correct answer and the results are actually meaningful.

Query design rationale
----------------------
Each query targets a different section of the paper and a different
retrieval challenge:

  Q1 — Scaled Dot-Product Attention (§3.2.1, p4)
       Definitional / formula lookup. Minimal vocabulary gap — a baseline
       where we expect HyDE to add little or no lift.

  Q2 — Why dot products are scaled by 1/√dk (§3.2.1, p4)
       Causal / "why" phrasing. The paper explains this in terms of
       "extremely small gradients" and variance of dk — a mild vocabulary
       gap between the colloquial question and the technical explanation.

  Q3 — How positional information is injected (§3.5, p6)
       Procedural. User says "positional information"; paper says
       "positional encodings" / "sine and cosine functions". Small but
       real vocabulary drift — a fair HyDE test.

  Q4 — Training hardware and duration (§5.2, p7)
       Factual / statistical lookup. User asks colloquially; the answer
       is a specific table of GPU counts and step counts. BM25 should
       excel here; HyDE is expected to be neutral or slightly negative.

  Q5 — Multi-head vs single-head attention trade-off (§3.2.2 + Table 3)
       Comparative / analytical. Requires synthesising the motivation in
       §3.2.2 with the ablation data in Table 3 row (A). Moderate
       vocabulary gap: "trade-off" and "benefit" vs "jointly attend" and
       "averaging inhibits".

Usage:
    python -m backend.scripts.hyde_comparison

Prerequisites: ChromaDB must be populated with pdftest1.pdf.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
os.environ.setdefault("GCS_BUCKET_NAME", "placeholder-bucket")

from loguru import logger

import backend.config as config
from backend.services import embeddings, vectorstore
from backend.services.retrieval import hyde_expand

# ---------------------------------------------------------------------------
# Test queries — all answerable from "Attention Is All You Need" (pdftest1.pdf)
# ---------------------------------------------------------------------------
# Each entry is (query_string, expected_page_hint) so the printout can flag
# whether the top result landed on the right page.
TEST_QUERIES: list[tuple[str, int]] = [
    (
        # Q1 — Definitional, minimal vocabulary gap.
        # Correct answer: p4 (§3.2.1 — Scaled Dot-Product Attention formula).
        "How is Scaled Dot-Product Attention computed?",
        4,
    ),
    (
        # Q2 — Causal "why", mild vocabulary gap.
        # Correct answer: p4 (scaling by 1/√dk prevents small gradients).
        "Why are the dot products divided by the square root of the key dimension?",
        4,
    ),
    (
        # Q3 — Procedural, small vocabulary drift ("positional information" vs
        # "positional encodings" / "sine and cosine").
        # Correct answer: p6 (§3.5 — Positional Encoding).
        "How does the Transformer encode positional information in the input?",
        6,
    ),
    (
        # Q4 — Factual / statistical lookup, colloquial phrasing.
        # Correct answer: p7 (§5.2 — Hardware and Schedule: 8 P100 GPUs, 100K steps).
        "How long did it take to train the base Transformer model and on what hardware?",
        7,
    ),
    (
        # Q5 — Comparative / analytical, moderate vocabulary gap.
        # Correct answer: p5 (§3.2.2) + p9 (Table 3 row A).
        # User says "benefit"; paper says "jointly attend to information from
        # different representation subspaces" and ablation shows single-head degrades.
        "What is the benefit of using multiple attention heads instead of one?",
        5,
    ),
]

TOP_K: int = 3


def _truncate(text: str, max_chars: int = 120) -> str:
    return text[:max_chars].replace("\n", " ") + ("…" if len(text) > max_chars else "")


def _run_vector_search(query_text: str) -> list[dict]:
    """Embed query_text and return top-K vector results."""
    embedding = embeddings.embed_text(query_text)
    return vectorstore.search(embedding, top_k=TOP_K)


def _page_tag(page: int, expected: int) -> str:
    """Return a ✓ or ✗ indicator comparing retrieved page to expected."""
    return "✓" if page == expected else f"✗ (expected p{expected})"


def compare_query(query: str, expected_page: int) -> None:
    """Print a side-by-side HyDE vs raw-query vector comparison for one query."""
    print(f"\n{'═' * 80}")
    print(f"QUERY: {query}")
    print(f"Expected answer page: p{expected_page}")
    print("═" * 80)

    # ---- Generate hypothetical document ----
    hypothetical = hyde_expand(query)
    print(f"\n[HyDE doc]\n  {_truncate(hypothetical, 200)}\n")

    # ---- Vector search: HyDE embedding ----
    hyde_results = _run_vector_search(hypothetical)

    # ---- Vector search: raw query embedding ----
    raw_results = _run_vector_search(query)

    # ---- Build text→rank maps for change detection ----
    raw_rank_map: dict[str, int] = {r["text"]: i + 1 for i, r in enumerate(raw_results)}
    hyde_rank_map: dict[str, int] = {r["text"]: i + 1 for i, r in enumerate(hyde_results)}

    # ---- Print raw results ----
    print("  WITHOUT HyDE (raw query embedding):")
    for i, r in enumerate(raw_results, 1):
        hyde_rank = hyde_rank_map.get(r["text"])
        if hyde_rank is None:
            change = "  [dropped from HyDE top-3]"
        elif hyde_rank < i:
            change = f"  ↑ rank {i}→{hyde_rank} with HyDE"
        elif hyde_rank > i:
            change = f"  ↓ rank {i}→{hyde_rank} with HyDE"
        else:
            change = "  = same rank with HyDE"
        src = r.get("source", "?")
        pg = r.get("page", "?")
        score = r.get("score", 0.0)
        page_ok = _page_tag(pg, expected_page) if i == 1 else ""
        print(f"    #{i} [{score:.4f}] {src} p{pg} {page_ok}| {_truncate(r['text'])}{change}")

    # ---- Print HyDE results ----
    print("\n  WITH HyDE (hypothetical doc embedding):")
    for i, r in enumerate(hyde_results, 1):
        raw_rank = raw_rank_map.get(r["text"])
        if raw_rank is None:
            change = "  [new entry — not in raw top-3]"
        elif raw_rank > i:
            change = f"  ↑ rank {raw_rank}→{i} (promoted by HyDE)"
        elif raw_rank < i:
            change = f"  ↓ rank {raw_rank}→{i} (demoted by HyDE)"
        else:
            change = "  = same rank as raw"
        src = r.get("source", "?")
        pg = r.get("page", "?")
        score = r.get("score", 0.0)
        page_ok = _page_tag(pg, expected_page) if i == 1 else ""
        print(f"    #{i} [{score:.4f}] {src} p{pg} {page_ok}| {_truncate(r['text'])}{change}")

    # ---- Summary ----
    raw_texts = {r["text"] for r in raw_results}
    hyde_texts = {r["text"] for r in hyde_results}
    new_in_hyde = hyde_texts - raw_texts
    dropped = raw_texts - hyde_texts

    raw_top1_page = raw_results[0].get("page") if raw_results else None
    hyde_top1_page = hyde_results[0].get("page") if hyde_results else None
    raw_hit = "✓ hit" if raw_top1_page == expected_page else "✗ miss"
    hyde_hit = "✓ hit" if hyde_top1_page == expected_page else "✗ miss"

    print(
        f"\n  Summary: {len(new_in_hyde)} new chunk(s) surfaced by HyDE, "
        f"{len(dropped)} dropped from raw top-3.\n"
        f"  Top-1 relevance — raw: {raw_hit} (p{raw_top1_page})  |  "
        f"HyDE: {hyde_hit} (p{hyde_top1_page})"
    )


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{level}: {message}")

    print("HyDE A/B Comparison — vector search only (BM25 excluded)")
    print(f"Model: {config.GENERATION_MODEL}  |  Embedding: {config.EMBEDDING_MODEL}")
    print(f"Corpus: pdftest1.pdf ('Attention Is All You Need', Vaswani et al. 2017)")
    print(f"Top-K per run: {TOP_K}")
    print(
        "\nNote: All queries are scoped to content in the indexed corpus.\n"
        "  ✓ = top-1 result landed on the expected page\n"
        "  ✗ = top-1 result missed (check the HyDE doc for vocabulary drift)\n"
    )

    for query, expected_page in TEST_QUERIES:
        try:
            compare_query(query, expected_page)
        except Exception as exc:
            logger.error(f"Failed for query {query!r}: {exc}")

    print(f"\n{'═' * 80}")
    print("Done. ✓/✗ indicators show whether each method found the right page at rank 1.")
    print("A consistent ✗ for HyDE where raw scores ✓ is evidence of HyDE-induced drift.")
    print("A consistent ✓ for HyDE where raw scores ✗ is evidence of genuine HyDE lift.")


if __name__ == "__main__":
    main()
