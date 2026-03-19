"""
Query logging service — persists per-query telemetry to SQLite.

Every successful call to POST /query writes one row to the query_logs table
in the same DB file used by the eval pipeline. The table is created lazily on
first write so no migration step is needed.
"""

import sqlite3
from pathlib import Path

from loguru import logger

import backend.config as config


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS query_logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,
    query_text          TEXT    NOT NULL,
    latency_ms          REAL    NOT NULL,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    retrieval_strategy  TEXT    NOT NULL DEFAULT 'hybrid',
    reranker_used       INTEGER NOT NULL DEFAULT 1
)
"""

_INSERT_SQL = """
INSERT INTO query_logs
    (timestamp, query_text, latency_ms, input_tokens, output_tokens,
     retrieval_strategy, reranker_used)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_query(
    *,
    timestamp: str,
    query_text: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    retrieval_strategy: str = "hybrid",
    reranker_used: bool = True,
) -> None:
    """Insert one row into query_logs, creating the table if needed.

    Failures are logged but never raised — telemetry must not break queries.
    """
    db_path = Path(config.EVAL_DB_PATH)
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(
                _INSERT_SQL,
                (
                    timestamp,
                    query_text,
                    latency_ms,
                    input_tokens,
                    output_tokens,
                    retrieval_strategy,
                    int(reranker_used),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        logger.error(f"query_logger: failed to write log row: {exc}")
