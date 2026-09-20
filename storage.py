"""Single-table DuckDB persistence for experiment snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb


COLUMNS = [
    "snapshot_id", "observed_at", "asset", "market_slug", "market_id",
    "condition_id", "window_start", "window_end", "rules", "resolution_source",
    "up_token_id", "down_token_id", "up_best_bid", "up_best_ask",
    "down_best_bid", "down_best_ask", "up_last_trade_price",
    "down_last_trade_price", "p_market", "proxy_source", "proxy_start_price",
    "proxy_current_price", "distance_from_start_bps", "return_1m", "return_5m",
    "realized_vol_5m", "realized_vol_15m", "time_remaining_sec", "raw_market_json",
    "checkpoint", "p_simple", "outcome", "resolved_at",
    "p_jev_blind", "p_jev_meta", "jev_model", "jev_blind_latency_ms",
    "jev_meta_latency_ms", "jev_blind_input_tokens", "jev_meta_input_tokens",
]


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id UUID PRIMARY KEY,
    observed_at TIMESTAMPTZ NOT NULL,
    asset VARCHAR NOT NULL,
    market_slug VARCHAR NOT NULL,
    market_id VARCHAR NOT NULL,
    condition_id VARCHAR NOT NULL,
    window_start BIGINT NOT NULL,
    window_end BIGINT NOT NULL,
    rules VARCHAR NOT NULL,
    resolution_source VARCHAR NOT NULL,
    up_token_id VARCHAR NOT NULL,
    down_token_id VARCHAR NOT NULL,
    up_best_bid DOUBLE,
    up_best_ask DOUBLE,
    down_best_bid DOUBLE,
    down_best_ask DOUBLE,
    up_last_trade_price DOUBLE,
    down_last_trade_price DOUBLE,
    p_market DOUBLE,
    proxy_source VARCHAR NOT NULL,
    proxy_start_price DOUBLE NOT NULL,
    proxy_current_price DOUBLE NOT NULL,
    distance_from_start_bps DOUBLE NOT NULL,
    return_1m DOUBLE NOT NULL,
    return_5m DOUBLE NOT NULL,
    realized_vol_5m DOUBLE NOT NULL,
    realized_vol_15m DOUBLE NOT NULL,
    time_remaining_sec DOUBLE NOT NULL,
    raw_market_json VARCHAR NOT NULL,
    checkpoint VARCHAR,
    p_simple DOUBLE,
    outcome VARCHAR,
    resolved_at TIMESTAMPTZ,
    p_jev_blind DOUBLE,
    p_jev_meta DOUBLE,
    jev_model VARCHAR,
    jev_blind_latency_ms DOUBLE,
    jev_meta_latency_ms DOUBLE,
    jev_blind_input_tokens BIGINT,
    jev_meta_input_tokens BIGINT
)
"""


UPGRADE_COLUMNS = {
    "checkpoint": "VARCHAR",
    "p_simple": "DOUBLE",
    "outcome": "VARCHAR",
    "resolved_at": "TIMESTAMPTZ",
    "p_jev_blind": "DOUBLE",
    "p_jev_meta": "DOUBLE",
    "jev_model": "VARCHAR",
    "jev_blind_latency_ms": "DOUBLE",
    "jev_meta_latency_ms": "DOUBLE",
    "jev_blind_input_tokens": "BIGINT",
    "jev_meta_input_tokens": "BIGINT",
}


def _ensure_schema(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(CREATE_SQL)
    existing = {row[1] for row in connection.execute("PRAGMA table_info('snapshots')").fetchall()}
    for name, data_type in UPGRADE_COLUMNS.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE snapshots ADD COLUMN {name} {data_type}")


def initialize_database(database_path: str) -> None:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(path)) as connection:
        _ensure_schema(connection)


def save_snapshot(database_path: str, snapshot: dict[str, Any]) -> None:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [name for name in COLUMNS if name not in snapshot]
    if missing:
        raise RuntimeError(f"cannot save snapshot: missing {', '.join(missing)}")
    placeholders = ", ".join("?" for _ in COLUMNS)
    with duckdb.connect(str(path)) as connection:
        _ensure_schema(connection)
        connection.execute(
            f"INSERT INTO snapshots ({', '.join(COLUMNS)}) VALUES ({placeholders})",
            [snapshot[name] for name in COLUMNS],
        )


def snapshot_exists(database_path: str, market_slug: str, checkpoint: str) -> bool:
    initialize_database(database_path)
    with duckdb.connect(database_path) as connection:
        result = connection.execute(
            "SELECT 1 FROM snapshots WHERE market_slug = ? AND checkpoint = ? LIMIT 1",
            [market_slug, checkpoint],
        ).fetchone()
    return result is not None


def unresolved_markets(database_path: str, now_timestamp: float) -> list[dict[str, Any]]:
    initialize_database(database_path)
    with duckdb.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT market_slug, asset, window_start, window_end
            FROM snapshots
            WHERE outcome IS NULL AND window_end < ?
            GROUP BY market_slug, asset, window_start, window_end
            ORDER BY window_end
            """,
            [now_timestamp],
        ).fetchall()
    return [
        {"market_slug": row[0], "asset": row[1], "window_start": row[2], "window_end": row[3]}
        for row in rows
    ]


def set_market_outcome(
    database_path: str, market_slug: str, outcome: str, resolved_at: Any
) -> None:
    initialize_database(database_path)
    with duckdb.connect(database_path) as connection:
        connection.execute(
            "UPDATE snapshots SET outcome = ?, resolved_at = ? WHERE market_slug = ?",
            [outcome, resolved_at, market_slug],
        )


def update_jev_results(
    database_path: str,
    snapshot_id: str,
    p_jev_blind: float,
    p_jev_meta: float,
    jev_model: str | None,
    jev_blind_latency_ms: float,
    jev_meta_latency_ms: float,
    jev_blind_input_tokens: int | None,
    jev_meta_input_tokens: int | None,
) -> None:
    initialize_database(database_path)
    with duckdb.connect(database_path) as connection:
        exists = connection.execute(
            "SELECT 1 FROM snapshots WHERE snapshot_id = ?", [snapshot_id]
        ).fetchone()
        if exists is None:
            raise RuntimeError(f"snapshot not found for Jev update: {snapshot_id}")
        connection.execute(
            """
            UPDATE snapshots SET
                p_jev_blind = ?, p_jev_meta = ?, jev_model = ?,
                jev_blind_latency_ms = ?, jev_meta_latency_ms = ?,
                jev_blind_input_tokens = ?, jev_meta_input_tokens = ?
            WHERE snapshot_id = ?
            """,
            [
                p_jev_blind,
                p_jev_meta,
                jev_model,
                jev_blind_latency_ms,
                jev_meta_latency_ms,
                jev_blind_input_tokens,
                jev_meta_input_tokens,
                snapshot_id,
            ],
        )


def status_summary(database_path: str) -> dict[str, Any] | None:
    path = Path(database_path)
    if not path.exists():
        return None
    initialize_database(database_path)
    with duckdb.connect(database_path) as connection:
        total = connection.execute("SELECT count(*) FROM snapshots").fetchone()[0]
        if total == 0:
            return None
        counts = connection.execute(
            """
            SELECT
                count(DISTINCT market_slug),
                count(DISTINCT CASE WHEN outcome IS NOT NULL THEN market_slug END),
                count(*) FILTER (WHERE asset = 'BTC'),
                count(*) FILTER (WHERE asset = 'ETH'),
                count(*) FILTER (WHERE checkpoint = 'T-10'),
                count(*) FILTER (WHERE checkpoint = 'T-5'),
                count(*) FILTER (WHERE checkpoint = 'T-2'),
                count(DISTINCT CASE WHEN outcome IS NULL THEN market_slug END),
                count(*) FILTER (
                    WHERE checkpoint IS NOT NULL
                      AND p_jev_blind IS NOT NULL AND p_jev_meta IS NOT NULL
                ),
                count(*) FILTER (
                    WHERE checkpoint IS NOT NULL
                      AND (p_jev_blind IS NULL OR p_jev_meta IS NULL)
                )
            FROM snapshots
            """
        ).fetchone()
        last = connection.execute(
            """
            SELECT CAST(observed_at AS VARCHAR), asset, market_slug, checkpoint, time_remaining_sec,
                   p_market, p_simple
            FROM snapshots ORDER BY observed_at DESC LIMIT 1
            """
        ).fetchone()
    return {
        "markets": counts[0],
        "snapshots": total,
        "resolved": counts[1],
        "btc": counts[2],
        "eth": counts[3],
        "t10": counts[4],
        "t5": counts[5],
        "t2": counts[6],
        "unresolved": counts[7],
        "jev_complete": counts[8],
        "jev_missing": counts[9],
        "last": last,
    }
