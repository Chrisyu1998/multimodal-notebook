"""
Eval API router.

Endpoints:
    GET /eval/runs           — List all eval runs (summary metadata from SQLite).
    GET /eval/runs/{run_id}  — Return full results for one run (from JSON file).
    GET /eval/latest         — Return full results for the most recent run.
"""

import json
import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger

import backend.config as config
from backend.models.schemas import EvalRunDetail, EvalRunMeta, EvalQueryResult

router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _list_runs_from_db() -> list[EvalRunMeta]:
    """Return all rows from eval_runs ordered newest-first."""
    db_path = Path(config.EVAL_DB_PATH)
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT run_id, timestamp, dataset_version, num_queries,
                   avg_latency_ms, p95_latency_ms, failed_queries,
                   avg_correctness, avg_hallucination_rate,
                   avg_faithfulness, avg_context_precision
            FROM eval_runs
            ORDER BY timestamp DESC
            """
        ).fetchall()
        return [EvalRunMeta(**dict(row)) for row in rows]
    finally:
        conn.close()


def _timestamp_to_filename(timestamp: str) -> str:
    """Reconstruct the JSON filename from a run's ISO timestamp."""
    ts_compact = (
        timestamp
        .replace(":", "-")
        .replace(".", "-")
        .replace("+", "")
    )
    return f"results_{ts_compact}.json"


def _load_run_json(run_id: str, timestamp: str) -> dict:
    """Load the full run JSON for a given run_id / timestamp pair."""
    results_dir = Path(config.EVAL_RESULTS_DIR)
    # Try the canonical filename first.
    candidate = results_dir / _timestamp_to_filename(timestamp)
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # Fall back to scanning all JSON files (handles clock-skew edge cases).
    for path in results_dir.glob("results_*.json"):
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("run_id") == run_id:
                return data
        except (json.JSONDecodeError, KeyError):
            continue

    raise FileNotFoundError(f"No JSON file found for run_id={run_id}")


def _parse_run_detail(data: dict) -> EvalRunDetail:
    """Convert a raw run dict into an EvalRunDetail Pydantic model."""
    results = [
        EvalQueryResult(
            query_id=r.get("query_id", ""),
            query=r.get("query", ""),
            category=r.get("category", ""),
            source_modality=r.get("source_modality", ""),
            status=r.get("status", "error"),
            generated_answer=r.get("generated_answer", ""),
            ground_truth=r.get("ground_truth", ""),
            latency_ms=r.get("latency_ms", 0.0),
            input_tokens=r.get("input_tokens", 0),
            output_tokens=r.get("output_tokens", 0),
            scores=r.get("scores"),
            reasoning=r.get("reasoning"),
        )
        for r in data.get("results", [])
    ]
    return EvalRunDetail(
        run_id=data["run_id"],
        timestamp=data["timestamp"],
        dataset_version=data.get("dataset_version", ""),
        results=results,
        summary=data.get("summary", {}),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/runs", response_model=list[EvalRunMeta])
async def list_runs() -> list[EvalRunMeta]:
    """Return all eval runs ordered newest-first."""
    runs = _list_runs_from_db()
    logger.info(f"Returning {len(runs)} eval run(s) from DB.")
    return runs


@router.get("/latest", response_model=EvalRunDetail)
async def get_latest_run() -> EvalRunDetail:
    """Return the full results for the most recent eval run."""
    runs = _list_runs_from_db()
    if not runs:
        raise HTTPException(status_code=404, detail="No eval runs found.")
    latest = runs[0]
    try:
        data = _load_run_json(latest.run_id, latest.timestamp)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        raise HTTPException(status_code=404, detail=str(exc))
    return _parse_run_detail(data)


@router.get("/runs/{run_id}", response_model=EvalRunDetail)
async def get_run(run_id: str) -> EvalRunDetail:
    """Return the full results for a specific eval run by run_id."""
    runs = _list_runs_from_db()
    meta = next((r for r in runs if r.run_id == run_id), None)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    try:
        data = _load_run_json(meta.run_id, meta.timestamp)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        raise HTTPException(status_code=404, detail=str(exc))
    return _parse_run_detail(data)
