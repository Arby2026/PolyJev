"""Single-table DuckDB persistence for Stage 1 snapshots."""

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
    raw_market_json VARCHAR NOT NULL
)
"""


def save_snapshot(database_path: str, snapshot: dict[str, Any]) -> None:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [name for name in COLUMNS if name not in snapshot]
    if missing:
        raise RuntimeError(f"cannot save snapshot: missing {', '.join(missing)}")
    placeholders = ", ".join("?" for _ in COLUMNS)
    with duckdb.connect(str(path)) as connection:
        connection.execute(CREATE_SQL)
        connection.execute(
            f"INSERT INTO snapshots ({', '.join(COLUMNS)}) VALUES ({placeholders})",
            [snapshot[name] for name in COLUMNS],
        )

