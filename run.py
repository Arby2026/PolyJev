"""Stage 1 CLI: collect and persist one Polymarket/Binance snapshot."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from features import calculate_features
from market_data import fetch_binance_data
from polymarket import best_prices, fetch_market, fetch_order_book, window_start_for_timestamp
from storage import save_snapshot


def load_config() -> dict[str, Any]:
    path = Path(__file__).with_name("config.yaml")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise RuntimeError("config.yaml is malformed")
    return config


def create_session(retries: int) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=0.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "PolyJev-stage1/1.0"
    return session


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("CLOB malformed response: invalid last trade price") from exc


def _price(value: float | None, digits: int = 4) -> str:
    return "NULL" if value is None else f"{value:.{digits}f}"


def print_summary(snapshot: dict[str, Any]) -> None:
    start = datetime.fromtimestamp(snapshot["window_start"], timezone.utc)
    end = datetime.fromtimestamp(snapshot["window_end"], timezone.utc)
    asset = snapshot["asset"]
    print(f"{asset} Up or Down 15m")
    print(f"slug: {snapshot['market_slug']}")
    print(f"window: {start:%H:%M:%S} -> {end:%H:%M:%S} UTC")
    print(f"time remaining: {int(snapshot['time_remaining_sec'])} sec")
    print("\nResolution:")
    print(f"Chainlink {asset}/USD TWAP")
    print(snapshot["resolution_source"])
    print("\nPolymarket:")
    print(f"UP    bid {_price(snapshot['up_best_bid'])} | ask {_price(snapshot['up_best_ask'])}")
    print(f"DOWN  bid {_price(snapshot['down_best_bid'])} | ask {_price(snapshot['down_best_ask'])}")
    print(f"P_market: {_price(snapshot['p_market'])}")
    print("\nUnderlying proxy (Binance Spot; not the resolution price):")
    print(f"window start: {snapshot['proxy_start_price']:.2f}")
    print(f"current:      {snapshot['proxy_current_price']:.2f}")
    print("\nFeatures:")
    print(f"distance_from_start_bps: {snapshot['distance_from_start_bps']:+.2f}")
    print(f"return_1m: {snapshot['return_1m']:+.6%}")
    print(f"return_5m: {snapshot['return_5m']:+.6%}")
    print(f"realized_vol_5m: {snapshot['realized_vol_5m']:.6%}")
    print(f"realized_vol_15m: {snapshot['realized_vol_15m']:.6%}")
    print(f"\nsaved snapshot: {snapshot['snapshot_id']}")
    print(f"database: {snapshot['database_path']}")


def collect_snapshot(asset: str, explicit_window_start: int | None) -> dict[str, Any]:
    config = load_config()
    timeout = float(config["http_timeout_sec"])
    session = create_session(int(config["http_retries"]))
    discovery_time = datetime.now(timezone.utc).timestamp()
    requested_start = (
        explicit_window_start
        if explicit_window_start is not None
        else window_start_for_timestamp(discovery_time)
    )
    market_data = fetch_market(
        session=session,
        gamma_base_url=config["gamma_base_url"],
        asset=asset,
        requested_start=requested_start,
        timeout=timeout,
        allow_fallback=explicit_window_start is None,
        now_timestamp=discovery_time,
    )
    if datetime.now(timezone.utc).timestamp() > market_data["window_end"]:
        raise RuntimeError("snapshot time is after window_end")
    market = market_data["market"]
    up_book = fetch_order_book(
        session, config["clob_base_url"], market_data["tokens"]["Up"], timeout
    )
    down_book = fetch_order_book(
        session, config["clob_base_url"], market_data["tokens"]["Down"], timeout
    )
    up_best_bid, up_best_ask = best_prices(up_book)
    down_best_bid, down_best_ask = best_prices(down_book)
    p_market = (
        (up_best_bid + up_best_ask) / 2
        if up_best_bid is not None and up_best_ask is not None
        else None
    )

    observed_at = datetime.now(timezone.utc)
    if observed_at.timestamp() > market_data["window_end"]:
        raise RuntimeError("snapshot time is after window_end")
    current_price, candles = fetch_binance_data(
        session=session,
        base_url=config["binance_base_url"],
        asset=asset,
        window_start=market_data["window_start"],
        observed_at_timestamp=observed_at.timestamp(),
        timeout=timeout,
    )
    feature_values = calculate_features(
        candles,
        current_price,
        market_data["window_start"],
        market_data["window_end"],
        observed_at.timestamp(),
    )
    database_path = str(config["database_path"])
    snapshot: dict[str, Any] = {
        "snapshot_id": str(uuid.uuid4()),
        "observed_at": observed_at,
        "asset": asset,
        "market_slug": market_data["slug"],
        "market_id": str(market.get("id") or ""),
        "condition_id": str(market.get("conditionId") or ""),
        "window_start": market_data["window_start"],
        "window_end": market_data["window_end"],
        "rules": market_data["rules"],
        "resolution_source": market_data["resolution_source"],
        "up_token_id": market_data["tokens"]["Up"],
        "down_token_id": market_data["tokens"]["Down"],
        "up_best_bid": up_best_bid,
        "up_best_ask": up_best_ask,
        "down_best_bid": down_best_bid,
        "down_best_ask": down_best_ask,
        "up_last_trade_price": _optional_float(
            up_book.get("last_trade_price", up_book.get("lastTradePrice"))
        ),
        "down_last_trade_price": _optional_float(
            down_book.get("last_trade_price", down_book.get("lastTradePrice"))
        ),
        "p_market": p_market,
        "proxy_source": "Binance Spot REST (predictive proxy only)",
        **feature_values,
        "raw_market_json": json.dumps(market, ensure_ascii=False, separators=(",", ":")),
    }
    if not snapshot["market_id"] or not snapshot["condition_id"]:
        raise RuntimeError("Gamma malformed response: missing market id or condition id")
    save_snapshot(database_path, snapshot)
    snapshot["database_path"] = database_path
    return snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jev x Polymarket Stage 1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot", help="capture one market snapshot")
    snapshot.add_argument("--asset", required=True, choices=("BTC", "ETH"))
    snapshot.add_argument("--window-start", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.window_start is not None and args.window_start % 900 != 0:
            raise RuntimeError("window-start must be aligned to a 15-minute UTC boundary")
        snapshot = collect_snapshot(args.asset, args.window_start)
        print_summary(snapshot)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
