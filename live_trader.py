"""Fee-aware realtime Jev x Polymarket paper trader."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from dotenv import load_dotenv
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from websockets.asyncio.client import connect

if not os.getenv("OPENROUTER_API_KEY", "").strip() and Path(".env").exists():
    os.environ.pop("OPENROUTER_API_KEY", None)
    load_dotenv(dotenv_path=Path(".env"))

from jev import call_jev_choice_async, create_jev_client
from live_market import LiveMarketState, iter_messages
from paper_trader import FeeSchedule, PaperTrader, Side, TransitionResult
from polymarket import (
    fetch_market,
    parse_resolved_outcome,
    window_start_for_timestamp,
)


WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
JEV_MODEL = "~typesafe/jev-latest"
DEFAULT_NOTIONAL = Decimal("10")
RESOLUTION_WAIT_SEC = 60
RESOLUTION_POLL_SEC = 5
CHOICE_INSTRUCTIONS = (
    "You are controlling a paper trader for the current Polymarket 15-minute "
    "crypto Up/Down market. Choose the TARGET POSITION: UP, DOWN, or FLAT. "
    "All fee and transition-cost fields were calculated deterministically by "
    "code using the market's current Polymarket fee schedule. Do not recalculate "
    "fees. Optimize net trading outcome after execution costs. A model decision "
    "is executed only after inference finishes, using the then-current executable "
    "order book, so avoid marginal edges that can be erased by latency or spread. "
    "Return only the typed Choice decision."
)
CHOICE_CRITERIA = {
    "UP": (
        "Target a LONG UP position now. Choose this only if, after considering "
        "the current executable prices, spreads, taker fees, liquidity, recent "
        "order-book/trade dynamics, current position, and time remaining, UP "
        "offers a better net trading opportunity than DOWN or FLAT."
    ),
    "DOWN": (
        "Target a LONG DOWN position now. Choose this only if, after considering "
        "the current executable prices, spreads, taker fees, liquidity, recent "
        "order-book/trade dynamics, current position, and time remaining, DOWN "
        "offers a better net trading opportunity than UP or FLAT."
    ),
    "FLAT": (
        "Target no position. Choose this if neither side offers enough expected "
        "edge to justify its executable spread and taker fees, or if closing the "
        "current position is preferable to remaining exposed or flipping."
    ),
}


@dataclass(frozen=True)
class MarketInfo:
    asset: str
    slug: str
    window_start: int
    window_end: int
    rules: str
    resolution_source: str
    up_token: str
    down_token: str
    minimum_tick_size: Decimal
    minimum_order_size: Decimal
    fee_schedule: FeeSchedule
    raw_market: dict[str, Any]


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, record_type: str, **values: Any) -> None:
        payload = {"record_type": record_type, **values}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class DecisionScheduler:
    """Runs at most one decision and coalesces updates to the latest dirty state."""

    def __init__(
        self,
        decide: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
        on_result: Callable[
            [dict[str, Any], dict[str, Any], datetime, datetime], Awaitable[None]
        ],
        on_error: Callable[[Exception], Awaitable[None]] | None = None,
        can_start: Callable[[], bool] | None = None,
    ):
        self.decide = decide
        self.on_result = on_result
        self.on_error = on_error
        self.can_start = can_start or (lambda: True)
        self.latest: dict[str, Any] | None = None
        self.latest_fingerprint: str | None = None
        self.last_processed_fingerprint: str | None = None
        self.dirty = False
        self.task: asyncio.Task[None] | None = None
        self.in_flight = False
        self.started = 0
        self.completed = 0
        self.errors = 0

    def submit(self, payload: dict[str, Any]) -> bool:
        fingerprint = state_fingerprint(payload["jev_state"])
        if fingerprint == self.latest_fingerprint:
            return False
        if self.task is None and fingerprint == self.last_processed_fingerprint:
            return False
        self.latest = copy.deepcopy(payload)
        self.latest_fingerprint = fingerprint
        self.dirty = True
        if self.task is None:
            self.task = asyncio.create_task(self._run())
        return True

    async def _run(self) -> None:
        try:
            while self.dirty:
                self.dirty = False
                payload = copy.deepcopy(self.latest)
                fingerprint = self.latest_fingerprint
                if payload is None or fingerprint == self.last_processed_fingerprint:
                    continue
                if not self.can_start():
                    self.dirty = False
                    break
                self.in_flight = True
                self.started += 1
                started_at = datetime.now(timezone.utc)
                try:
                    result = await self.decide(payload)
                    finished_at = datetime.now(timezone.utc)
                    self.completed += 1
                    await self.on_result(payload, result, started_at, finished_at)
                except Exception as exc:
                    self.errors += 1
                    if self.on_error is not None:
                        await self.on_error(exc)
                finally:
                    self.in_flight = False
                    self.last_processed_fingerprint = fingerprint
        finally:
            self.task = None

    async def wait_idle(self) -> None:
        if self.task is not None:
            await self.task


def state_fingerprint(state: Mapping[str, Any]) -> str:
    canonical = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def discover_current_market(asset: str, notional: Decimal) -> tuple[MarketInfo, Any, dict[str, Any]]:
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise RuntimeError(
            "Установите OPENROUTER_API_KEY в окружении Windows (Окружения) "
            "или в .env, см .env.example"
        )
    config = _load_config()
    session = _create_http_session(int(config["http_retries"]))
    now = time.time()
    window_start = window_start_for_timestamp(now)
    market_data = fetch_market(
        session,
        str(config["gamma_base_url"]),
        asset,
        window_start,
        float(config["http_timeout_sec"]),
        False,
        now,
    )
    market = market_data["market"]
    required_status = {
        "active": market.get("active") is True,
        "not closed": market.get("closed") is False,
        "accepting orders": market.get("acceptingOrders") is True,
        "order book enabled": market.get("enableOrderBook") is True,
    }
    failed = [name for name, valid in required_status.items() if not valid]
    if failed:
        raise RuntimeError("market is not trade-ready: " + ", ".join(failed))
    try:
        tick_size = Decimal(str(market["orderPriceMinTickSize"]))
        minimum_order_size = Decimal(str(market["orderMinSize"]))
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError("market is missing trading constraints") from exc
    if notional < minimum_order_size:
        raise RuntimeError(
            f"notional {notional} is below minimum order size {minimum_order_size}"
        )
    fee_schedule = parse_fee_schedule(market)
    info = MarketInfo(
        asset=asset,
        slug=market_data["slug"],
        window_start=market_data["window_start"],
        window_end=market_data["window_end"],
        rules=market_data["rules"],
        resolution_source=market_data["resolution_source"],
        up_token=market_data["tokens"]["Up"],
        down_token=market_data["tokens"]["Down"],
        minimum_tick_size=tick_size,
        minimum_order_size=minimum_order_size,
        fee_schedule=fee_schedule,
        raw_market=market,
    )
    return info, session, config


def parse_fee_schedule(market: Mapping[str, Any]) -> FeeSchedule:
    enabled = market.get("feesEnabled")
    if enabled is False:
        return FeeSchedule(False)
    raw = market.get("feeSchedule")
    if enabled is not True or not isinstance(raw, Mapping):
        raise RuntimeError("unsupported fee schedule for paper simulation")
    try:
        schedule = FeeSchedule(
            True,
            Decimal(str(raw["rate"])),
            Decimal(str(raw["exponent"])),
            raw["takerOnly"] is True,
            Decimal(str(raw["rebateRate"])),
        )
        schedule.validate()
        return schedule
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError("unsupported fee schedule for paper simulation") from exc


class LivePaperSession:
    def __init__(self, market: MarketInfo, notional: Decimal, session: Any, config: dict[str, Any]):
        self.market = market
        self.http_session = session
        self.config = config
        self.state = LiveMarketState(market.up_token, market.down_token)
        self.state.tick_size = market.minimum_tick_size
        self.paper = PaperTrader(notional, market.fee_schedule)
        self.started_at = datetime.now(timezone.utc)
        self.log = JsonlWriter(
            Path(__file__).parent
            / "data"
            / f"live_{market.asset.lower()}_{market.window_start}.jsonl"
        )
        self.client = create_jev_client()
        self.market_events = 0
        self.meaningful_updates = 0
        self.busy_updates = 0
        self.input_tokens = 0
        self.latencies: list[float] = []
        self.settlement_pending = True
        self.scheduler = DecisionScheduler(
            self._decide,
            self._handle_decision,
            self._handle_jev_error,
            lambda: time.time() < self.market.window_end
            and self.state.resolved_winner is None,
        )

    async def _decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await call_jev_choice_async(
            self.client,
            JEV_MODEL,
            payload["jev_state"],
            CHOICE_INSTRUCTIONS,
            CHOICE_CRITERIA,
        )

    async def _handle_jev_error(self, exc: Exception) -> None:
        print(f"Jev error: {exc}", file=sys.stderr)

    def build_decision_payload(self) -> dict[str, Any]:
        books = self.state.execution_books()
        mark = self.paper.mark_to_market(books)
        position = {
            "target": self.paper.position.side,
            "shares": str(self.paper.position.shares),
            "entry_cost": str(self.paper.position.total_entry_cost),
            "liquidation_value_net": _optional_decimal(mark["liquidation_value_net"]),
            "unrealized_pnl_net": _optional_decimal(mark["unrealized_pnl"]),
        }
        summary = self.state.summary()
        jev_state = {
            "market": {
                "asset": self.market.asset,
                "time_remaining_sec": max(0, int(self.market.window_end - time.time())),
                "resolution_rules": self.market.rules,
                "resolution_source": self.market.resolution_source,
                "minimum_order_size": str(self.market.minimum_order_size),
                "tick_size": str(self.state.tick_size),
            },
            "fees": {
                "enabled": self.market.fee_schedule.enabled,
                "taker_fee_rate": str(self.market.fee_schedule.rate),
                "taker_only": self.market.fee_schedule.taker_only,
                "note": "All execution-cost fields below are already fee-adjusted.",
            },
            "up": summary["up"],
            "down": summary["down"],
            "position": position,
            "target_transitions": self.paper.preview_transitions(books),
        }
        return {
            "jev_state": jev_state,
            "captured_monotonic": time.perf_counter(),
            "input_state_summary": quote_summary(self.state),
            "decision_quotes": jev_state["target_transitions"],
        }

    async def _handle_decision(
        self,
        payload: dict[str, Any],
        decision: dict[str, Any],
        started_at: datetime,
        finished_at: datetime,
    ) -> None:
        self.latencies.append(float(decision["latency_ms"]))
        self.input_tokens += int(decision.get("input_tokens") or 0)
        current_books = self.state.execution_books()
        execution_summary = {
            "book": quote_summary(self.state),
            "target_transitions": self.paper.preview_transitions(current_books),
        }
        transition = execute_choice_on_latest_state(
            self.paper,
            decision["choice"],
            self.state,
            time.time() >= self.market.window_end
            or self.state.resolved_winner is not None,
        )
        mark = self.paper.mark_to_market(self.state.execution_books())
        decision_id = str(uuid.uuid4())
        record = {
            "decision_id": decision_id,
            "decision_started_at": started_at.isoformat(),
            "decision_finished_at": finished_at.isoformat(),
            "latency_ms": decision["latency_ms"],
            "state_age_ms": (time.perf_counter() - payload["captured_monotonic"]) * 1000,
            "market_slug": self.market.slug,
            "time_remaining_at_input": payload["jev_state"]["market"]["time_remaining_sec"],
            "input_state_summary": payload["input_state_summary"],
            "choice": decision["choice"],
            "choice_probabilities": decision["probabilities"],
            "position_before": transition.position_before,
            "decision_quotes": payload["decision_quotes"],
            "execution_quotes": execution_summary,
            "executed": transition.executed,
            "execution_block_reason": transition.block_reason,
            "fills": [fill.as_dict() for fill in transition.fills],
            "fees_paid": str(transition.fees_paid),
            "position_after": transition.position_after,
            "realized_pnl": str(self.paper.realized_trading_pnl),
            "unrealized_pnl": _optional_decimal(mark["unrealized_pnl"]),
            "input_tokens": decision.get("input_tokens"),
        }
        self.log.write("decision", **record)
        print_decision(self.market, self.state, self.paper, decision, transition, mark)
        if (
            self.scheduler.dirty
            and self.state.trade_ready()
            and time.time() < self.market.window_end
        ):
            self.scheduler.submit(self.build_decision_payload())

    async def handle_message(self, payload: Any) -> None:
        for message in iter_messages(payload):
            self.market_events += 1
            meaningful = self.state.apply_event(message)
            if not meaningful:
                continue
            self.meaningful_updates += 1
            if self.scheduler.in_flight:
                self.busy_updates += 1
            if self.state.trade_ready() and time.time() < self.market.window_end:
                self.scheduler.submit(self.build_decision_payload())

    async def run(self) -> None:
        self._print_startup()
        self.log.write(
            "session_start",
            started_at=self.started_at.isoformat(),
            market_slug=self.market.slug,
            asset=self.market.asset,
            window_start=self.market.window_start,
            window_end=self.market.window_end,
            notional=str(self.paper.notional),
            fee_schedule=fee_schedule_dict(self.market.fee_schedule),
        )
        resolution_deadline = self.market.window_end + RESOLUTION_WAIT_SEC
        next_resolution_poll = self.market.window_end
        while time.time() < resolution_deadline and self.state.resolved_winner is None:
            try:
                self.state.reset_for_reconnect()
                async with connect(WS_URL, ping_interval=None, close_timeout=2) as websocket:
                    await websocket.send(
                        json.dumps(
                            {
                                "assets_ids": [self.market.up_token, self.market.down_token],
                                "type": "market",
                                "custom_feature_enabled": True,
                            }
                        )
                    )
                    print("WebSocket connected")
                    heartbeat = asyncio.create_task(self._heartbeat(websocket))
                    try:
                        while (
                            time.time() < resolution_deadline
                            and self.state.resolved_winner is None
                        ):
                            try:
                                raw = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            except asyncio.TimeoutError:
                                raw = None
                            if raw == "PONG":
                                continue
                            if raw is not None:
                                try:
                                    payload = json.loads(raw)
                                    await self.handle_message(payload)
                                except (ValueError, TypeError) as exc:
                                    print(f"WebSocket message ignored: {exc}", file=sys.stderr)
                            now = time.time()
                            if now >= self.market.window_end and now >= next_resolution_poll:
                                winner = await asyncio.to_thread(self._poll_resolution)
                                next_resolution_poll = now + RESOLUTION_POLL_SEC
                                if winner is not None:
                                    self.state.resolved_winner = winner
                    finally:
                        heartbeat.cancel()
                        await asyncio.gather(heartbeat, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if time.time() >= resolution_deadline:
                    break
                print(f"WebSocket disconnected: {exc}; reconnecting", file=sys.stderr)
                await asyncio.sleep(1)
        await self.scheduler.wait_idle()
        winner = self.state.resolved_winner
        if winner is not None:
            settlement = self.paper.settle(winner)
            self.settlement_pending = False
            self.log.write(
                "market_resolved",
                resolved_at=datetime.now(timezone.utc).isoformat(),
                winner=winner,
                settlement_pnl=str(settlement),
            )
        self._print_summary()
        self.client.__exit__(None, None, None)

    async def _heartbeat(self, websocket: Any) -> None:
        while True:
            await asyncio.sleep(10)
            await websocket.send("PING")

    def _poll_resolution(self) -> str | None:
        fresh = fetch_market(
            self.http_session,
            str(self.config["gamma_base_url"]),
            self.market.asset,
            self.market.window_start,
            float(self.config["http_timeout_sec"]),
            False,
            time.time(),
        )
        market = fresh["market"]
        if market.get("closed") is not True and fresh["event"].get("closed") is not True:
            return None
        return parse_resolved_outcome(market.get("outcomes"), market.get("outcomePrices"))

    def _print_startup(self) -> None:
        fee = self.market.fee_schedule
        print("Realtime Jev Polymarket Paper Trader")
        print("\nMarket:")
        print(f"slug: {self.market.slug}")
        print(f"time remaining: {max(0, int(self.market.window_end - time.time()))} sec")
        print("\nTrading:")
        print(f"min tick: {self.market.minimum_tick_size}")
        print(f"min order: {self.market.minimum_order_size}")
        print("\nFees:")
        print(f"enabled: {str(fee.enabled).lower()}")
        print(f"rate: {fee.rate}")
        print(f"exponent: {fee.exponent}")
        print(f"taker_only: {str(fee.taker_only).lower()}")
        print(f"rebate_rate: {fee.rebate_rate}\n")

    def _print_summary(self) -> None:
        duration = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        mark = self.paper.mark_to_market(self.state.execution_books())
        unrealized = mark["unrealized_pnl"] or Decimal("0")
        total = self.paper.total_pnl + unrealized
        summary = {
            "market_slug": self.market.slug,
            "asset": self.market.asset,
            "session_duration_sec": duration,
            "market_events": self.market_events,
            "meaningful_updates": self.meaningful_updates,
            "jev_decisions_started": self.scheduler.started,
            "jev_decisions_completed": self.scheduler.completed,
            "busy_updates": self.busy_updates,
            "jev_errors": self.scheduler.errors,
            "trades": self.paper.trades,
            "opens": self.paper.opens,
            "closes": self.paper.closes,
            "flips": self.paper.flips,
            "avg_jev_latency_ms": avg_latency,
            "latest_jev_latency_ms": self.latencies[-1] if self.latencies else None,
            "input_tokens": self.input_tokens,
            "total_taker_fees": str(self.paper.total_fees),
            "realized_trading_pnl": str(self.paper.realized_trading_pnl),
            "settlement_pnl": str(self.paper.settlement_pnl),
            "unrealized_pnl": str(unrealized),
            "total_pnl": str(total),
            "settlement_pending": self.settlement_pending,
            "trade_history": self.paper.trade_history,
        }
        print("\nSession summary")
        print("---------------")
        for key, value in summary.items():
            if key != "trade_history":
                print(f"{key}: {value}")
        print("trade_history:")
        for trade in self.paper.trade_history:
            print(
                f"  {trade['kind']} {trade['from']} -> {trade['to']} | "
                f"fees {trade['fees']} | realized {trade['realized_pnl']}"
            )
        self.log.write("session_summary", **summary)


def blocked_transition(choice: Side, paper: PaperTrader, reason: str) -> TransitionResult:
    position = paper.position.snapshot()
    return TransitionResult(
        choice, position, position.copy(), False, reason, (), Decimal("0"), Decimal("0"), "none"
    )


def execute_choice_on_latest_state(
    paper: PaperTrader,
    choice: Side,
    state: LiveMarketState,
    market_ended: bool,
) -> TransitionResult:
    if market_ended:
        return blocked_transition(choice, paper, "decision discarded: market ended")
    if not state.trade_ready():
        return blocked_transition(choice, paper, "execution state is not trade-ready")
    return paper.execute_target(choice, state.execution_books())


def quote_summary(state: LiveMarketState) -> dict[str, Any]:
    summary = state.summary()
    return {
        "up": {
            "best_bid": summary["up"]["best_bid"],
            "best_ask": summary["up"]["best_ask"],
            "best_bid_size": summary["up"]["best_bid_size"],
            "best_ask_size": summary["up"]["best_ask_size"],
        },
        "down": {
            "best_bid": summary["down"]["best_bid"],
            "best_ask": summary["down"]["best_ask"],
            "best_bid_size": summary["down"]["best_bid_size"],
            "best_ask_size": summary["down"]["best_ask_size"],
        },
    }


def print_decision(
    market: MarketInfo,
    state: LiveMarketState,
    paper: PaperTrader,
    decision: Mapping[str, Any],
    transition: TransitionResult,
    mark: Mapping[str, Decimal | None],
) -> None:
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    left = max(0, int(market.window_end - time.time()))
    probabilities = decision["probabilities"]
    print(
        f"\n{now} | left {left // 60:02}:{left % 60:02}\n"
        f"UP {state.up.best_bid}/{state.up.best_ask} | "
        f"DOWN {state.down.best_bid}/{state.down.best_ask}\n"
        f"Jev -> {decision['choice']} [UP {probabilities['UP']:.2f} "
        f"DOWN {probabilities['DOWN']:.2f} FLAT {probabilities['FLAT']:.2f}]\n"
        f"latency {decision['latency_ms']:.0f}ms\n"
        f"position {paper.position.side} | uPnL {_optional_decimal(mark['unrealized_pnl'])}"
    )
    if transition.executed:
        print(f"TRADE\n{transition.position_before['side']} -> {transition.target}")
        for fill in transition.fills:
            cash = (
                fill.gross_value + fill.fee
                if fill.action == "BUY"
                else fill.gross_value - fill.fee
            )
            cash_label = "cash out" if fill.action == "BUY" else "cash in"
            print(
                f"{fill.action} {fill.shares} @ {fill.price} | gross {fill.gross_value} "
                f"| fee {fill.fee} | {cash_label} {cash}"
            )
    elif transition.block_reason != "target unchanged":
        print(transition.block_reason)


def fee_schedule_dict(schedule: FeeSchedule) -> dict[str, Any]:
    return {
        "enabled": schedule.enabled,
        "rate": str(schedule.rate),
        "exponent": str(schedule.exponent),
        "taker_only": schedule.taker_only,
        "rebate_rate": str(schedule.rebate_rate),
    }


def _optional_decimal(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _load_config() -> dict[str, Any]:
    with Path(__file__).with_name("config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise RuntimeError("config.yaml is malformed")
    return config


def _create_http_session(retries: int) -> requests.Session:
    session = requests.Session()
    policy = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=0.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=policy))
    session.headers["User-Agent"] = "PolyJev-live-paper/1.0"
    return session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime Jev Polymarket paper trader")
    parser.add_argument("--asset", required=True, choices=("BTC", "ETH"))
    parser.add_argument("--notional", type=Decimal, default=DEFAULT_NOTIONAL)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.notional <= Decimal("0"):
            raise RuntimeError("notional must be positive")
        market, session, config = discover_current_market(args.asset, args.notional)
        asyncio.run(LivePaperSession(market, args.notional, session, config).run())
        return 0
    except KeyboardInterrupt:
        print("paper trader stopped")
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
