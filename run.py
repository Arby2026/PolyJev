"""CLI for manual snapshots and the Stage 2 autonomous collector."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from features import calculate_features, calculate_p_simple
from jev import (
    BLIND_INSTRUCTIONS,
    META_INSTRUCTIONS,
    build_blind_state,
    build_meta_state,
    call_jev,
    create_jev_client,
)
from market_data import fetch_binance_data
from polymarket import (
    best_prices,
    fetch_market,
    fetch_order_book,
    market_slug,
    parse_resolved_outcome,
    window_start_for_timestamp,
)
from storage import (
    initialize_database,
    save_snapshot,
    set_market_outcome,
    snapshot_exists,
    status_summary,
    unresolved_markets,
    update_jev_results,
)


CHECKPOINTS = {"T-10": 600, "T-5": 300, "T-2": 120}
CHECKPOINT_GRACE_SEC = 15
POLL_INTERVAL_SEC = 2
RESOLUTION_INTERVAL_SEC = 30


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
    session.headers["User-Agent"] = "PolyJev-stage2/1.0"
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


def checkpoint_due(
    time_remaining_sec: float, target_sec: int, grace_sec: int = CHECKPOINT_GRACE_SEC
) -> bool:
    return target_sec - grace_sec <= time_remaining_sec <= target_sec


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
    print(f"P_simple (Binance proxy): {_price(snapshot['p_simple'])}")
    print("\nUnderlying proxy (Binance Spot; not the resolution price):")
    print(f"window start: {snapshot['proxy_start_price']:.2f}")
    print(f"current:      {snapshot['proxy_current_price']:.2f}")
    print("\nFeatures:")
    print(f"distance_from_start_bps: {snapshot['distance_from_start_bps']:+.2f}")
    print(f"return_1m: {snapshot['return_1m']:+.6%}")
    print(f"return_5m: {snapshot['return_5m']:+.6%}")
    print(f"realized_vol_5m: {snapshot['realized_vol_5m']:.6%}")
    print(f"realized_vol_15m: {snapshot['realized_vol_15m']:.6%}")
    if snapshot["p_jev_blind"] is not None and snapshot["p_jev_meta"] is not None:
        print("\nJev:")
        print(f"Blind P(UP): {snapshot['p_jev_blind']:.4f}")
        print(f"Meta  P(UP): {snapshot['p_jev_meta']:.4f}")
        print(f"Model: {snapshot['jev_model']}")
        print(f"Blind latency: {snapshot['jev_blind_latency_ms']:.1f} ms")
        print(f"Meta latency: {snapshot['jev_meta_latency_ms']:.1f} ms")
    print(f"\nsaved snapshot: {snapshot['snapshot_id']}")
    print(f"database: {snapshot['database_path']}")


def collect_snapshot(
    asset: str, explicit_window_start: int | None, checkpoint: str | None = None
) -> dict[str, Any] | None:
    config = load_config()
    timeout = float(config["http_timeout_sec"])
    session = create_session(int(config["http_retries"]))
    discovery_time = datetime.now(timezone.utc).timestamp()
    requested_start = (
        explicit_window_start
        if explicit_window_start is not None
        else window_start_for_timestamp(discovery_time)
    )
    database_path = str(config["database_path"])
    expected_slug = market_slug(asset, requested_start)
    if checkpoint is not None:
        if checkpoint not in CHECKPOINTS:
            raise RuntimeError(f"unknown checkpoint: {checkpoint}")
        if snapshot_exists(database_path, expected_slug, checkpoint):
            return None
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
    if checkpoint is not None and not checkpoint_due(
        feature_values["time_remaining_sec"], CHECKPOINTS[checkpoint]
    ):
        raise RuntimeError(f"{checkpoint} checkpoint window was missed during collection")
    p_simple = calculate_p_simple(
        feature_values["proxy_start_price"],
        feature_values["proxy_current_price"],
        feature_values["realized_vol_15m"],
        feature_values["time_remaining_sec"],
    )
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
        "checkpoint": checkpoint,
        "p_simple": p_simple,
        "outcome": None,
        "resolved_at": None,
        "p_jev_blind": None,
        "p_jev_meta": None,
        "jev_model": None,
        "jev_blind_latency_ms": None,
        "jev_meta_latency_ms": None,
        "jev_blind_input_tokens": None,
        "jev_meta_input_tokens": None,
    }
    if not snapshot["market_id"] or not snapshot["condition_id"]:
        raise RuntimeError("Gamma malformed response: missing market id or condition id")
    save_snapshot(database_path, snapshot)
    snapshot["database_path"] = database_path
    return snapshot


def _require_jev_api_key() -> None:
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise RuntimeError("OPENROUTER_API_KEY is not set")


def _enrich_snapshot_with_jev(
    client: Any, snapshot: dict[str, Any], config: dict[str, Any]
) -> None:
    model = str(config["jev_model"])
    blind = call_jev(client, model, build_blind_state(snapshot), BLIND_INSTRUCTIONS)
    meta = call_jev(client, model, build_meta_state(snapshot), META_INSTRUCTIONS)
    model = meta["model"] or blind["model"] or str(config["jev_model"])
    try:
        update_jev_results(
            str(config["database_path"]),
            snapshot["snapshot_id"],
            blind["probability"],
            meta["probability"],
            model,
            blind["latency_ms"],
            meta["latency_ms"],
            blind["input_tokens"],
            meta["input_tokens"],
        )
    except Exception as exc:
        raise RuntimeError(f"failed to save Jev results: {exc}") from exc
    snapshot.update(
        {
            "p_jev_blind": blind["probability"],
            "p_jev_meta": meta["probability"],
            "jev_model": model,
            "jev_blind_latency_ms": blind["latency_ms"],
            "jev_meta_latency_ms": meta["latency_ms"],
            "jev_blind_input_tokens": blind["input_tokens"],
            "jev_meta_input_tokens": meta["input_tokens"],
        }
    )


def _collector_status_path(database_path: str) -> Path:
    return Path(database_path).with_name("collector_status.json")


def _write_last_error(database_path: str, message: str | None) -> None:
    path = _collector_status_path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_error": message,
        "at": datetime.now(timezone.utc).isoformat() if message else None,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _read_last_error(database_path: str) -> str:
    path = _collector_status_path(database_path)
    if not path.exists():
        return "none"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "status file is unreadable"
    message = payload.get("last_error")
    if not message:
        return "none"
    observed_at = payload.get("at")
    return f"{observed_at}: {message}" if observed_at else str(message)


def _resolve_finished_markets(config: dict[str, Any]) -> tuple[list[tuple[str, str, str]], list[str]]:
    database_path = str(config["database_path"])
    now = datetime.now(timezone.utc)
    pending = unresolved_markets(database_path, now.timestamp())
    if not pending:
        return [], []
    session = create_session(int(config["http_retries"]))
    resolved: list[tuple[str, str, str]] = []
    errors: list[str] = []
    for item in pending:
        try:
            fresh = fetch_market(
                session=session,
                gamma_base_url=config["gamma_base_url"],
                asset=item["asset"],
                requested_start=item["window_start"],
                timeout=float(config["http_timeout_sec"]),
                allow_fallback=False,
                now_timestamp=now.timestamp(),
            )
            market = fresh["market"]
            event = fresh["event"]
            if market.get("closed") is not True and event.get("closed") is not True:
                continue
            outcome = parse_resolved_outcome(
                market.get("outcomes"), market.get("outcomePrices")
            )
            if outcome is None:
                continue
            resolved_at = datetime.now(timezone.utc)
            set_market_outcome(database_path, item["market_slug"], outcome, resolved_at)
            resolved.append((item["asset"], item["market_slug"], outcome))
        except RuntimeError as exc:
            errors.append(f"[{item['asset']}] resolution error: {exc}")
    return resolved, errors


def run_collector(with_jev: bool = False) -> int:
    config = load_config()
    database_path = str(config["database_path"])
    initialize_database(database_path)
    _write_last_error(database_path, None)
    jev_client = None
    if with_jev:
        _require_jev_api_key()
        jev_client = create_jev_client()
    print("PolyJev collector started")
    print("assets: BTC, ETH")
    print("checkpoints: T-10, T-5, T-2")
    last_resolution_check = 0.0
    try:
        while True:
            loop_time = datetime.now(timezone.utc).timestamp()
            window_start = window_start_for_timestamp(loop_time)
            window_end = window_start + 900
            time_remaining = window_end - loop_time
            for checkpoint, target_sec in CHECKPOINTS.items():
                if not checkpoint_due(time_remaining, target_sec):
                    continue
                captured_snapshots: list[dict[str, Any]] = []
                for asset in ("BTC", "ETH"):
                    slug = market_slug(asset, window_start)
                    if snapshot_exists(database_path, slug, checkpoint):
                        continue
                    try:
                        snapshot = collect_snapshot(asset, window_start, checkpoint)
                        if snapshot is not None:
                            captured_snapshots.append(snapshot)
                            print(
                                f"[{asset}] {checkpoint} saved {snapshot['market_slug']} "
                                f"({snapshot['time_remaining_sec']:.1f}s remaining)"
                            )
                    except RuntimeError as exc:
                        message = f"[{asset}] {checkpoint} error: {exc}"
                        print(message, file=sys.stderr)
                        _write_last_error(database_path, message)
                if jev_client is not None:
                    for snapshot in captured_snapshots:
                        try:
                            _enrich_snapshot_with_jev(jev_client, snapshot, config)
                            print(
                                f"[{snapshot['asset']}] {checkpoint} Jev enriched "
                                f"(blind={snapshot['p_jev_blind']:.4f}, "
                                f"meta={snapshot['p_jev_meta']:.4f})"
                            )
                        except RuntimeError as exc:
                            message = (
                                f"[{snapshot['asset']}] {checkpoint} Jev error: {exc}"
                            )
                            print(message, file=sys.stderr)
                            _write_last_error(database_path, message)

            if loop_time - last_resolution_check >= RESOLUTION_INTERVAL_SEC:
                resolved, errors = _resolve_finished_markets(config)
                for asset, _slug, outcome in resolved:
                    print(f"[{asset}] market resolved: {outcome}")
                for message in errors:
                    print(message, file=sys.stderr)
                    _write_last_error(database_path, message)
                last_resolution_check = loop_time
            time.sleep(POLL_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("PolyJev collector stopped")
        return 0
    finally:
        if jev_client is not None:
            jev_client.__exit__(None, None, None)


def print_status() -> None:
    config = load_config()
    database_path = str(config["database_path"])
    summary = status_summary(database_path)
    if summary is None:
        print(f"No snapshots in {database_path}")
        print(f"Last error: {_read_last_error(database_path)}")
        return
    print(f"Markets observed: {summary['markets']}")
    print(f"Snapshots: {summary['snapshots']}")
    print(f"Resolved markets: {summary['resolved']}")
    print()
    print(f"BTC snapshots: {summary['btc']}")
    print(f"ETH snapshots: {summary['eth']}")
    print()
    print(f"T-10: {summary['t10']}")
    print(f"T-5: {summary['t5']}")
    print(f"T-2: {summary['t2']}")
    print()
    print(f"Unresolved: {summary['unresolved']}")
    print(f"Jev complete snapshots: {summary['jev_complete']}")
    print(f"Jev missing snapshots: {summary['jev_missing']}")
    observed_at, asset, slug, checkpoint, remaining, p_market, p_simple = summary["last"]
    print("\nLast snapshot:")
    print(f"{observed_at} | {asset} | {slug} | {checkpoint or 'manual'}")
    print(
        f"time_remaining={remaining:.1f}s | p_market={_price(p_market)} | "
        f"p_simple={_price(p_simple)}"
    )
    print(f"\nLast error: {_read_last_error(database_path)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jev x Polymarket")
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot", help="capture one market snapshot")
    snapshot.add_argument("--asset", required=True, choices=("BTC", "ETH"))
    snapshot.add_argument("--window-start", type=int)
    snapshot.add_argument("--with-jev", action="store_true")
    collect = subparsers.add_parser("collect", help="run the autonomous checkpoint collector")
    collect.add_argument("--with-jev", action="store_true")
    subparsers.add_parser("status", help="show local dataset status")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "snapshot":
            if args.window_start is not None and args.window_start % 900 != 0:
                raise RuntimeError("window-start must be aligned to a 15-minute UTC boundary")
            snapshot = collect_snapshot(args.asset, args.window_start)
            if snapshot is not None:
                if args.with_jev:
                    _require_jev_api_key()
                    config = load_config()
                    with create_jev_client() as client:
                        _enrich_snapshot_with_jev(client, snapshot, config)
                print_summary(snapshot)
        elif args.command == "collect":
            return run_collector(args.with_jev)
        elif args.command == "status":
            print_status()
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
