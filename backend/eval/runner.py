"""
Eval runner — executes the full RAG pipeline against the golden dataset.

Pipeline per query: hybrid_search → rerank → generate_answer.
Results are written to {EVAL_RESULTS_DIR}/results_{timestamp}.json.
Run metadata is persisted to the SQLite eval_runs table at EVAL_DB_PATH.

Usage:
    python -m backend.eval.runner [--dry-run] [--category CATEGORY]

Flags:
    --dry-run     Run only the first 5 queries.
    --category    Filter by category (e.g. factual, multi-hop, cross-modal).
"""

import argparse
import asyncio
import json
import sqlite3
import time
from typing import Optional
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

import backend.config as config
from backend.services import bm25_index
from backend.services.generation import generate_answer
from backend.services.retrieval import hybrid_search, rerank

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATASET_VERSION: str = "1.0"
_DRY_RUN_LIMIT: int = 5
_CONCURRENCY_LIMIT: int = 5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: int) -> float:
    """Return the p-th percentile of *values* using linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lower = int(idx)
    upper = lower + 1
    if upper >= len(sorted_vals):
        return float(sorted_vals[-1])
    frac = idx - lower
    return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])


def _load_dataset(path: str, category: Optional[str]) -> list[dict]:
    """Load golden_dataset.json and optionally filter by category.

    Args:
        path:     Filesystem path to the golden dataset JSON file.
        category: If non-None, keep only queries whose 'category' matches.

    Returns:
        List of query dicts from the 'queries' array.
    """
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    queries: list[dict] = data["queries"]
    if category:
        queries = [q for q in queries if q.get("category") == category]
        logger.info(f"Category filter '{category}': {len(queries)} queries selected.")
    return queries


def _format_retrieved_chunks(chunks: list[dict]) -> list[dict]:
    """Serialize reranked chunks into the eval output schema.

    Uses rerank_score when available (post-reranking); falls back to the RRF
    score so the recorded value always reflects the final ranking signal.
    """
    return [
        {
            "text": chunk.get("text", ""),
            "source": chunk.get("source", ""),
            "page": chunk.get("page", 0),
            "modality": chunk.get("modality", ""),
            "score": round(
                float(chunk.get("rerank_score", chunk.get("score", 0.0))), 4
            ),
        }
        for chunk in chunks
    ]


def _compute_summary(results: list[dict]) -> dict:
    """Compute aggregate latency and failure statistics from per-query results."""
    failed: int = sum(
        1 for r in results if r["generated_answer"].startswith("ERROR:")
    )
    latencies: list[float] = [
        r["latency_ms"]
        for r in results
        if not r["generated_answer"].startswith("ERROR:")
    ]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
    p50 = round(_percentile(latencies, 50), 1) if latencies else 0.0
    p95 = round(_percentile(latencies, 95), 1) if latencies else 0.0
    return {
        "total_queries": len(results),
        "failed_queries": failed,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
    }


def _init_db(db_path: str) -> None:
    """Create the eval_runs table in the SQLite database if it does not exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_runs (
                run_id           TEXT PRIMARY KEY,
                timestamp        TEXT NOT NULL,
                dataset_version  TEXT NOT NULL,
                num_queries      INTEGER NOT NULL,
                avg_latency_ms   REAL NOT NULL,
                p95_latency_ms   REAL NOT NULL,
                failed_queries   INTEGER NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert_run_metadata(db_path: str, run_data: dict) -> None:
    """Write a single eval_runs row for the completed run."""
    summary = run_data["summary"]
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO eval_runs
              (run_id, timestamp, dataset_version, num_queries,
               avg_latency_ms, p95_latency_ms, failed_queries)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_data["run_id"],
                run_data["timestamp"],
                run_data["dataset_version"],
                summary["total_queries"],
                summary["avg_latency_ms"],
                summary["p95_latency_ms"],
                summary["failed_queries"],
            ),
        )
        conn.commit()
        logger.info(f"Run metadata written to {db_path} (run_id={run_data['run_id']})")
    finally:
        conn.close()


def _write_results(run_data: dict, results_dir: str) -> Path:
    """Serialize the full run dict to a timestamped JSON file and return its path."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Compact timestamp suitable for filenames: replace colons and periods.
    ts_compact = (
        run_data["timestamp"]
        .replace(":", "-")
        .replace(".", "-")
        .replace("+", "")
    )
    out_path = out_dir / f"results_{ts_compact}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(run_data, fh, indent=2, ensure_ascii=False)
    logger.info(f"Results written to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Per-query pipeline
# ---------------------------------------------------------------------------

async def _run_query(item: dict, semaphore: asyncio.Semaphore) -> dict:
    """Run the full RAG pipeline for one golden-dataset entry.

    Acquires the semaphore before calling into the retrieval/generation
    services so at most _CONCURRENCY_LIMIT queries are in-flight at once.
    All three service calls (hybrid_search, rerank, generate_answer) are
    synchronous and are offloaded to the thread pool via asyncio.to_thread
    to keep the event loop unblocked.

    On any exception the generated_answer is set to "ERROR: <message>" and
    the result is returned normally so the run continues for remaining queries.

    Args:
        item:      One entry from the golden dataset 'queries' array.
        semaphore: Limits concurrent in-flight pipeline calls.

    Returns:
        Per-query result dict matching the eval output schema.
    """
    query_id: str = item["id"]
    query: str = item["query"]
    category: str = item.get("category", "")
    source_modality: str = item.get("source_modality", "")
    ground_truth: str = item.get("ground_truth", "")

    async with semaphore:
        logger.info(f"Running query {query_id} [{category}]: {query!r}")
        start = time.monotonic()

        try:
            chunks: list[dict] = await asyncio.to_thread(
                hybrid_search, query, config.VECTOR_TOP_K
            )
            reranked: list[dict] = await asyncio.to_thread(
                rerank, query, chunks, config.RERANK_TOP_K
            )
            result: dict = await asyncio.to_thread(
                generate_answer, query, reranked
            )

            latency_ms = round((time.monotonic() - start) * 1000, 1)
            logger.info(
                f"Query {query_id} completed in {latency_ms:.1f}ms — "
                f"{result['chunks_used']} chunks used"
            )

            return {
                "query_id": query_id,
                "query": query,
                "category": category,
                "source_modality": source_modality,
                "generated_answer": result["answer"],
                "ground_truth": ground_truth,
                "retrieved_chunks": _format_retrieved_chunks(reranked),
                "latency_ms": latency_ms,
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
            }

        except Exception as exc:
            latency_ms = round((time.monotonic() - start) * 1000, 1)
            logger.error(
                f"Query {query_id} failed after {latency_ms:.1f}ms: {exc}"
            )
            return {
                "query_id": query_id,
                "query": query,
                "category": category,
                "source_modality": source_modality,
                "generated_answer": f"ERROR: {exc}",
                "ground_truth": ground_truth,
                "retrieved_chunks": [],
                "latency_ms": latency_ms,
                "input_tokens": 0,
                "output_tokens": 0,
            }


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

async def run_eval(queries: list[dict]) -> dict:
    """Execute the eval pipeline for all queries and return the full run dict.

    Schedules all queries as concurrent asyncio tasks, capped at
    _CONCURRENCY_LIMIT simultaneous API calls via asyncio.Semaphore.

    Args:
        queries: List of query dicts from the golden dataset.

    Returns:
        Complete run dict with keys: run_id, timestamp, dataset_version,
        results (one entry per query), summary (aggregate metrics).
    """
    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    semaphore = asyncio.Semaphore(_CONCURRENCY_LIMIT)

    logger.info(
        f"Eval run {run_id} starting — {len(queries)} queries, "
        f"concurrency={_CONCURRENCY_LIMIT}"
    )

    tasks = [_run_query(item, semaphore) for item in queries]
    results: list[dict] = await asyncio.gather(*tasks)

    summary = _compute_summary(results)
    logger.info(
        f"Eval run {run_id} complete — "
        f"{summary['total_queries']} queries, "
        f"{summary['failed_queries']} failed, "
        f"avg_latency={summary['avg_latency_ms']}ms "
        f"p95_latency={summary['p95_latency_ms']}ms"
    )

    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "dataset_version": _DATASET_VERSION,
        "results": results,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

async def main() -> None:
    """CLI entrypoint — parse flags, run eval, write outputs."""
    parser = argparse.ArgumentParser(
        description="Run the RAG eval pipeline against the golden dataset."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Run only the first {_DRY_RUN_LIMIT} queries.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter queries by category (factual, multi-hop, cross-modal, "
             "adversarial, out-of-scope).",
    )
    args = parser.parse_args()

    bm25_index.load_index()

    queries = _load_dataset(config.EVAL_DATASET_PATH, args.category)

    if args.dry_run:
        queries = queries[:_DRY_RUN_LIMIT]
        logger.info(f"Dry run mode: running {len(queries)} queries only.")

    if not queries:
        logger.warning("No queries to run — check --category filter.")
        return

    _init_db(config.EVAL_DB_PATH)

    run_data = await run_eval(queries)

    _write_results(run_data, config.EVAL_RESULTS_DIR)
    _insert_run_metadata(config.EVAL_DB_PATH, run_data)

    summary = run_data["summary"]
    logger.info(
        f"Done — avg={summary['avg_latency_ms']}ms "
        f"p95={summary['p95_latency_ms']}ms "
        f"failed={summary['failed_queries']}/{summary['total_queries']}"
    )


if __name__ == "__main__":
    asyncio.run(main())
