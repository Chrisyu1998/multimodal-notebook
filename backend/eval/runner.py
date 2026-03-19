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
from backend.eval.judge import (
    score_correctness,
    score_hallucination,
    score_faithfulness,
    score_context_precision,
)
from backend.services import bm25_index
from backend.services.generation import generate_answer
from backend.services.query_logger import log_query
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
    """Compute aggregate latency, token, failure, and judge score statistics."""
    failed: int = sum(1 for r in results if r.get("status") == "error")
    successful = [r for r in results if r.get("status") == "ok"]
    latencies: list[float] = [r["latency_ms"] for r in successful]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
    p50 = round(_percentile(latencies, 50), 1) if latencies else 0.0
    p95 = round(_percentile(latencies, 95), 1) if latencies else 0.0

    def _avg_score(key: str) -> float:
        vals = [r["scores"][key] for r in successful if r.get("scores")]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "total_queries": len(results),
        "failed_queries": failed,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "total_input_tokens": sum(r.get("input_tokens", 0) for r in results),
        "total_output_tokens": sum(r.get("output_tokens", 0) for r in results),
        "avg_correctness": _avg_score("correctness"),
        "avg_hallucination_rate": _avg_score("hallucination_rate"),
        "avg_faithfulness": _avg_score("faithfulness"),
        "avg_context_precision": _avg_score("context_precision"),
    }


def _init_db(db_path: str) -> None:
    """Create the eval_runs table in the SQLite database if it does not exist.

    Also migrates existing tables by adding judge score columns when absent.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_runs (
                run_id                TEXT PRIMARY KEY,
                timestamp             TEXT NOT NULL,
                dataset_version       TEXT NOT NULL,
                num_queries           INTEGER NOT NULL,
                avg_latency_ms        REAL NOT NULL,
                p95_latency_ms        REAL NOT NULL,
                failed_queries        INTEGER NOT NULL,
                avg_correctness       REAL NOT NULL DEFAULT 0.0,
                avg_hallucination_rate REAL NOT NULL DEFAULT 0.0,
                avg_faithfulness      REAL NOT NULL DEFAULT 0.0,
                avg_context_precision REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        # Migrate existing databases that predate the judge score columns.
        for col in (
            "avg_correctness",
            "avg_hallucination_rate",
            "avg_faithfulness",
            "avg_context_precision",
        ):
            try:
                conn.execute(
                    f"ALTER TABLE eval_runs ADD COLUMN {col} REAL NOT NULL DEFAULT 0.0"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists — expected on fresh runs.
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
               avg_latency_ms, p95_latency_ms, failed_queries,
               avg_correctness, avg_hallucination_rate,
               avg_faithfulness, avg_context_precision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_data["run_id"],
                run_data["timestamp"],
                run_data["dataset_version"],
                summary["total_queries"],
                summary["avg_latency_ms"],
                summary["p95_latency_ms"],
                summary["failed_queries"],
                summary["avg_correctness"],
                summary["avg_hallucination_rate"],
                summary["avg_faithfulness"],
                summary["avg_context_precision"],
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

    The RAG pipeline (hybrid_search → rerank → generate_answer) runs inside
    the semaphore so at most _CONCURRENCY_LIMIT queries are in-flight at once.
    After the semaphore is released, all four judge calls run concurrently via
    asyncio.gather — they are independent of the RAG services and do not need
    to be rate-limited by the same slot.

    On any pipeline exception, status="error" is returned immediately with
    scores=None so the run continues and _compute_summary can distinguish
    pipeline failures from judge failures without string-matching answers.

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

    # --- RAG pipeline (rate-limited by semaphore) ---
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
            generated_answer: str = result["answer"]
            logger.info(
                f"Query {query_id} completed in {latency_ms:.1f}ms — "
                f"{result['chunks_used']} chunks used"
            )
            log_query(
                timestamp=datetime.now(timezone.utc).isoformat(),
                query_text=query,
                latency_ms=latency_ms,
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                retrieval_strategy="hybrid",
                reranker_used=True,
            )
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
                "status": "error",
                "generated_answer": f"ERROR: {exc}",
                "ground_truth": ground_truth,
                "retrieved_chunks": [],
                "latency_ms": latency_ms,
                "input_tokens": 0,
                "output_tokens": 0,
                "scores": None,
                "reasoning": None,
            }

    # --- Judge scoring (concurrent, outside semaphore — independent of RAG services) ---
    try:
        correctness, hallucination, faithfulness, precision = await asyncio.gather(
            asyncio.to_thread(score_correctness, query, ground_truth, generated_answer, reranked),
            asyncio.to_thread(score_hallucination, query, ground_truth, generated_answer, reranked),
            asyncio.to_thread(score_faithfulness, query, ground_truth, generated_answer, reranked),
            asyncio.to_thread(score_context_precision, query, ground_truth, generated_answer, reranked),
        )
        logger.info(
            f"Query {query_id} judged — correctness={correctness['score']} "
            f"hallucination={hallucination['score']} "
            f"faithfulness={faithfulness['score']} "
            f"context_precision={precision['score']}"
        )
        scores = {
            "correctness": correctness["score"],
            "hallucination_rate": hallucination["score"],
            "faithfulness": faithfulness["score"],
            "context_precision": precision["score"],
        }
        reasoning = {
            "correctness": correctness["reasoning"],
            "hallucination": hallucination["reasoning"],
            "faithfulness": faithfulness["reasoning"],
            "context_precision": precision["reasoning"],
        }
    except Exception as exc:
        logger.error(f"Query {query_id} judge scoring failed: {exc}")
        scores = None
        reasoning = None

    return {
        "query_id": query_id,
        "query": query,
        "category": category,
        "source_modality": source_modality,
        "status": "ok",
        "generated_answer": generated_answer,
        "ground_truth": ground_truth,
        "retrieved_chunks": _format_retrieved_chunks(reranked),
        "latency_ms": latency_ms,
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "scores": scores,
        "reasoning": reasoning,
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
