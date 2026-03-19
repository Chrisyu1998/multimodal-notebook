"""
Metrics router — system-level query telemetry.

Endpoints:
    GET /metrics/timeseries  — Daily aggregates for the last 30 days.
    GET /metrics/summary     — All-time totals.
"""

import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter
from loguru import logger

import backend.config as config
from backend.models.schemas import MetricsSummary, TimeseriesPoint

router = APIRouter()

# Cost formula (Gemini 2.5 Flash pricing)
_INPUT_COST_PER_TOKEN: float = 0.00000125
_OUTPUT_COST_PER_TOKEN: float = 0.000005


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_rows(days: int) -> list[dict]:
    """Return query_log rows from the last *days* days, newest-first."""
    db_path = Path(config.EVAL_DB_PATH)
    if not db_path.exists():
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            # Table may not exist yet if no queries have been run.
            rows = conn.execute(
                """
                SELECT timestamp, query_text, latency_ms,
                       input_tokens, output_tokens,
                       retrieval_strategy, reranker_used
                FROM query_logs
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                """,
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except sqlite3.OperationalError:
        # Table doesn't exist yet — no queries logged.
        return []
    except Exception as exc:
        logger.error(f"metrics: DB read error: {exc}")
        return []


def _fetch_all_rows() -> list[dict]:
    """Return all rows from query_logs."""
    db_path = Path(config.EVAL_DB_PATH)
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT timestamp, latency_ms, input_tokens, output_tokens
                FROM query_logs
                ORDER BY timestamp DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return []
    except Exception as exc:
        logger.error(f"metrics: DB read error (all rows): {exc}")
        return []


def _pct(values: list[float], p: int) -> float:
    """Return the p-th percentile of *values* (0–100). Returns 0.0 for empty."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(len(sorted_vals) * p / 100) - 1)
    return sorted_vals[idx]


def _cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * _INPUT_COST_PER_TOKEN + output_tokens * _OUTPUT_COST_PER_TOKEN


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/timeseries", response_model=list[TimeseriesPoint])
async def get_timeseries() -> list[TimeseriesPoint]:
    """Return daily aggregates for the last 30 days, ordered oldest-first."""
    rows = _fetch_rows(days=30)

    # Group by ISO date (YYYY-MM-DD prefix of the timestamp).
    by_date: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        date_key = row["timestamp"][:10]
        by_date[date_key].append(row)

    points: list[TimeseriesPoint] = []
    for date_str, day_rows in sorted(by_date.items()):
        latencies = [r["latency_ms"] for r in day_rows]
        input_toks = [r["input_tokens"] for r in day_rows]
        output_toks = [r["output_tokens"] for r in day_rows]
        n = len(day_rows)
        total_input = sum(input_toks)
        total_output = sum(output_toks)
        points.append(
            TimeseriesPoint(
                date=date_str,
                avg_latency_ms=statistics.mean(latencies),
                p50_latency_ms=_pct(latencies, 50),
                p95_latency_ms=_pct(latencies, 95),
                total_queries=n,
                avg_input_tokens=total_input / n,
                avg_output_tokens=total_output / n,
                estimated_cost_usd=_cost(total_input, total_output),
            )
        )

    logger.info(f"metrics/timeseries: returning {len(points)} daily point(s)")
    return points


@router.get("/summary", response_model=MetricsSummary)
async def get_summary() -> MetricsSummary:
    """Return all-time query telemetry totals."""
    rows = _fetch_all_rows()

    if not rows:
        return MetricsSummary(
            total_queries=0,
            avg_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            avg_input_tokens=0.0,
            avg_output_tokens=0.0,
            total_cost_usd=0.0,
            avg_cost_per_query=0.0,
        )

    latencies = [r["latency_ms"] for r in rows]
    input_toks = [r["input_tokens"] for r in rows]
    output_toks = [r["output_tokens"] for r in rows]
    n = len(rows)
    total_input = sum(input_toks)
    total_output = sum(output_toks)
    total_cost = _cost(total_input, total_output)

    logger.info(f"metrics/summary: {n} total query log(s)")
    return MetricsSummary(
        total_queries=n,
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=_pct(latencies, 50),
        p95_latency_ms=_pct(latencies, 95),
        avg_input_tokens=total_input / n,
        avg_output_tokens=total_output / n,
        total_cost_usd=total_cost,
        avg_cost_per_query=total_cost / n,
    )
