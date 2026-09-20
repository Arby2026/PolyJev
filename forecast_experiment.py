"""Position-independent realtime Jev forecast experiment for one 15m market."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping

from websockets.asyncio.client import connect

from chainlink_live import (
    RAW_TOPIC,
    RTDS_URL,
    TWAP60_TOPIC,
    ChainlinkLiveState,
    subscription_frame,
)
from jev import call_jev_nouls_async, create_jev_client
from live_market import LiveMarketState, iter_messages
from live_trader import (
    DecisionScheduler,
    JsonlWriter,
    MarketInfo,
    WS_URL,
    _create_http_session,
    _load_config,
    parse_fee_schedule,
)
from paper_trader import (
    ExecutionQuote,
    FeeSchedule,
    Level,
    calculate_fee,
    quote_buy,
    quote_sell,
)
from polymarket import fetch_market, parse_resolved_outcome


JEV_MODEL = "~typesafe/jev-latest"
NOTIONAL = Decimal("10")
TOKEN_PRICE_USD = Decimal("0.042") / Decimal("1000000")
HORIZONS = (15, 30, 60, 120)
RULE_SUMMARY = "UP iff Chainlink TWAP for this contract range >= start price; else DOWN"
RESOLUTION_WAIT_SEC = 60
RESOLUTION_POLL_SEC = 5
STALE_BOOK_SEC = 2.0

RESOLVE_INSTRUCTION = "Will this contract resolve UP under the stated Chainlink rule?"


@dataclass(frozen=True)
class PendingLabel:
    decision_id: str
    side: str
    horizon: int
    probability: float
    entry_epoch: float
    target_epoch: float
    due_monotonic: float
    entry_quote: ExecutionQuote


@dataclass(frozen=True)
class LabelResult:
    status: str
    profitable: bool | None
    exit_quote: ExecutionQuote | None
    actual_net_pnl: Decimal | None
    delay_ms: float
    book_age_ms: float | None


class CostGuard:
    def __init__(self, maximum_usd: Decimal):
        if maximum_usd < 0:
            raise ValueError("max Jev cost must be non-negative")
        self.maximum_usd = maximum_usd
        self.input_tokens: list[int] = []
        self.estimated_cost = Decimal("0")
        self.exhausted_announced = False

    @property
    def can_request(self) -> bool:
        return self.estimated_cost < self.maximum_usd

    def record(self, input_tokens: int | None) -> Decimal:
        tokens = int(input_tokens or 0)
        self.input_tokens.append(tokens)
        request_cost = Decimal(tokens) * TOKEN_PRICE_USD
        self.estimated_cost += request_cost
        return request_cost

    def metrics(self) -> dict[str, Any]:
        values = self.input_tokens
        return {
            "input_tokens_total": sum(values),
            "input_tokens_avg": (sum(values) / len(values)) if values else 0,
            "input_tokens_p50": statistics.median(values) if values else 0,
            "input_tokens_max": max(values) if values else 0,
            "estimated_jev_cost": str(self.estimated_cost),
            "budget_exhausted": not self.can_request,
        }


def validate_resolution_rule(rules: str, resolution_source: str) -> str:
    text = f"{rules} {resolution_source}".casefold()
    comparison = "greater than or equal" in text or "higher than or equal" in text
    if "chainlink" not in text or "twap" not in text or not comparison:
        raise RuntimeError("unexpected market resolution rule")
    return RULE_SUMMARY


def discover_forecast_market(
    asset: str, window_start: int
) -> tuple[MarketInfo, Any, dict[str, Any]]:
    config = _load_config()
    session = _create_http_session(int(config["http_retries"]))
    data = fetch_market(
        session,
        str(config["gamma_base_url"]),
        asset,
        window_start,
        float(config["http_timeout_sec"]),
        False,
        time.time(),
    )
    market = data["market"]
    checks = {
        "active": market.get("active") is True,
        "not closed": market.get("closed") is False,
        "accepting orders": market.get("acceptingOrders") is True,
        "order book enabled": market.get("enableOrderBook") is True,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise RuntimeError("market is not forecast-ready: " + ", ".join(failed))
    rule = validate_resolution_rule(data["rules"], data["resolution_source"])
    try:
        tick = Decimal(str(market["orderPriceMinTickSize"]))
        minimum = Decimal(str(market["orderMinSize"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("market is missing trading constraints") from exc
    info = MarketInfo(
        asset=asset,
        slug=data["slug"],
        window_start=data["window_start"],
        window_end=data["window_end"],
        rules=rule,
        resolution_source=data["resolution_source"],
        up_token=data["tokens"]["Up"],
        down_token=data["tokens"]["Down"],
        minimum_tick_size=tick,
        minimum_order_size=minimum,
        fee_schedule=parse_fee_schedule(market),
        raw_market=market,
    )
    return info, session, config


def build_questions(
    time_remaining: int,
    entries: Mapping[str, ExecutionQuote],
) -> dict[str, str]:
    questions = {"resolve_up": RESOLVE_INSTRUCTION}
    for side in ("UP", "DOWN"):
        entry = entries.get(side)
        if entry is None or not entry.fully_executable:
            continue
        for horizon in HORIZONS:
            if time_remaining <= horizon + 2:
                continue
            name = f"{side.lower()}_profit_{horizon}s"
            questions[name] = (
                f"If $10 gross notional of {side} were bought now using the current "
                f"executable asks and Polymarket taker fees, will the full position be "
                f"sellable {horizon} seconds from now using executable bids with net PnL "
                "greater than zero after both entry and exit taker fees?"
            )
    return questions


def terminal_break_even(entry: ExecutionQuote) -> Decimal:
    if not entry.fully_executable or entry.shares <= 0:
        raise ValueError("terminal break-even requires an executable entry")
    return entry.net_cash / entry.shares


def terminal_edges(
    p_resolve_up: float,
    entries: Mapping[str, ExecutionQuote],
) -> dict[str, Decimal | None]:
    p_up = Decimal(str(p_resolve_up))
    result: dict[str, Decimal | None] = {}
    for side, probability in (("UP", p_up), ("DOWN", Decimal("1") - p_up)):
        entry = entries.get(side)
        key = side.lower()
        if entry is None or not entry.fully_executable:
            result[f"{key}_terminal_break_even"] = None
            result[f"model_terminal_edge_{key}"] = None
        else:
            break_even = terminal_break_even(entry)
            result[f"{key}_terminal_break_even"] = break_even
            result[f"model_terminal_edge_{key}"] = probability - break_even
    return result


def hold_to_resolution_pnl(entry: ExecutionQuote, side: str, winner: str) -> Decimal:
    proceeds = entry.shares if side == winner else Decimal("0")
    return proceeds - entry.net_cash


def break_even_exit_bid(entry: ExecutionQuote, schedule: FeeSchedule) -> Decimal | None:
    """Minimum uniform bid where exit net proceeds cover the entry cash cost."""
    target = entry.net_cash / entry.shares
    if target >= Decimal("1"):
        return None
    if not schedule.enabled:
        return target
    low, high = Decimal("0"), Decimal("1")
    for _ in range(80):
        middle = (low + high) / 2
        net_proceeds = (
            entry.shares * middle
            - calculate_fee(entry.shares, middle, schedule)
        )
        if net_proceeds > entry.net_cash:
            high = middle
        else:
            low = middle
    return high


def hypothetical_entries(
    state: LiveMarketState, schedule: FeeSchedule
) -> dict[str, ExecutionQuote]:
    books = state.execution_books()
    return {
        side: quote_buy(books[side]["asks"], NOTIONAL, schedule)
        for side in ("UP", "DOWN")
    }


def liquidity_snapshot(
    state: LiveMarketState,
    schedule: FeeSchedule,
    tick_size: Decimal,
) -> dict[str, Any]:
    books = state.execution_books()
    summary = state.summary()
    result: dict[str, Any] = {}
    for side, key in (("UP", "up"), ("DOWN", "down")):
        item = summary[key]
        entry = quote_buy(books[side]["asks"], NOTIONAL, schedule)
        exit_quote = (
            quote_sell(books[side]["bids"], entry.shares, schedule)
            if entry.fully_executable
            else None
        )
        bid_depth = sum((level.size for level in books[side]["bids"][:5]), Decimal("0"))
        ask_depth = sum((level.size for level in books[side]["asks"][:5]), Decimal("0"))
        result[key] = {
            "spread_ticks": str(Decimal(item["spread"]) / tick_size),
            "top5_bid_depth": str(bid_depth),
            "top5_ask_depth": str(ask_depth),
            "can_buy_10": entry.fully_executable,
            "immediate_roundtrip_executable": bool(
                exit_quote and exit_quote.fully_executable
            ),
        }
    return result


def build_compact_state(
    market: MarketInfo,
    chainlink: ChainlinkLiveState,
    clob: LiveMarketState,
    entries: Mapping[str, ExecutionQuote],
    now: float | None = None,
) -> dict[str, Any]:
    current_time = time.time() if now is None else now
    summary = clob.summary()
    state: dict[str, Any] = {
        "market": {
            "asset": market.asset,
            "left_s": max(0, int(market.window_end - current_time)),
            "rule": market.rules,
        },
        "chainlink": chainlink.compact(),
    }
    for side, key in (("UP", "up"), ("DOWN", "down")):
        item = summary[key]
        bid_depth = sum(
            (level.size for level in clob.by_token[
                market.up_token if side == "UP" else market.down_token
            ].top_levels("bids", 5)),
            Decimal("0"),
        )
        ask_depth = sum(
            (level.size for level in clob.by_token[
                market.up_token if side == "UP" else market.down_token
            ].top_levels("asks", 5)),
            Decimal("0"),
        )
        state[key] = {
            "bid": _model_decimal(item["best_bid"], market.minimum_tick_size),
            "ask": _model_decimal(item["best_ask"], market.minimum_tick_size),
            "spread_ticks": _model_decimal(
                Decimal(item["spread"]) / market.minimum_tick_size, Decimal("0.01")
            ),
            "imbalance": round(float(item["book_imbalance"]), 4),
            "depth_bid": _model_decimal(bid_depth, Decimal("0.01")),
            "depth_ask": _model_decimal(ask_depth, Decimal("0.01")),
            "mid_d1": _round_optional(item["mid_change_bps"]["last1"], 2),
            "mid_d5": _round_optional(item["mid_change_bps"]["last5"], 2),
            "mid_d20": _round_optional(item["mid_change_bps"]["last20"], 2),
            "last_trade_side": item["last_trade"]["side"],
        }
    economics: dict[str, Any] = {}
    for side in ("UP", "DOWN"):
        entry = entries.get(side)
        if entry is None or not entry.fully_executable:
            economics[side.lower()] = {"entry_executable": False}
            continue
        economics[side.lower()] = {
            "entry_executable": True,
            "entry_vwap": _model_decimal(entry.vwap, market.minimum_tick_size),
            "entry_fee": _model_decimal(entry.fee, Decimal("0.00001")),
            "terminal_break_even": _model_decimal(
                terminal_break_even(entry), Decimal("0.00001")
            ),
            "break_even_exit_bid": _model_decimal(
                break_even_exit_bid(entry, market.fee_schedule), Decimal("0.00001")
            ),
        }
    state["economics"] = economics
    return state


def evaluate_label(
    pending: PendingLabel,
    bids: list[Level],
    schedule: FeeSchedule,
    now_epoch: float,
    now_monotonic: float,
    last_book_update_monotonic: float,
    market_end: int,
) -> LabelResult:
    delay_ms = max(0.0, (now_epoch - pending.target_epoch) * 1000)
    book_age_ms = (
        (now_monotonic - last_book_update_monotonic) * 1000
        if last_book_update_monotonic > 0
        else None
    )
    if now_epoch >= market_end:
        return LabelResult("market_ended", None, None, None, delay_ms, book_age_ms)
    if book_age_ms is None or book_age_ms > STALE_BOOK_SEC * 1000:
        return LabelResult("stale", None, None, None, delay_ms, book_age_ms)
    exit_quote = quote_sell(bids, pending.entry_quote.shares, schedule)
    if not exit_quote.fully_executable:
        return LabelResult("unexecutable", None, exit_quote, None, delay_ms, book_age_ms)
    pnl = exit_quote.net_cash - pending.entry_quote.net_cash
    return LabelResult("valid", pnl > 0, exit_quote, pnl, delay_ms, book_age_ms)


def _model_decimal(value: Decimal | str | None, quantum: Decimal) -> str | None:
    if value is None:
        return None
    parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    text = format(parsed.quantize(quantum, rounding=ROUND_HALF_UP), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _round_optional(value: Any, digits: int) -> float | None:
    return None if value is None else round(float(value), digits)


class ForecastExperiment:
    def __init__(self, asset: str, max_jev_cost: Decimal):
        self.asset = asset
        self.symbol = "btc/usd" if asset == "BTC" else "eth/usd"
        self.chainlink = ChainlinkLiveState(self.symbol)
        self.cost = CostGuard(max_jev_cost)
        self.client = create_jev_client()
        self.stop = asyncio.Event()
        self.state_changed = asyncio.Event()
        self.market: MarketInfo | None = None
        self.http_session: Any = None
        self.config: dict[str, Any] | None = None
        self.clob: LiveMarketState | None = None
        self.log: JsonlWriter | None = None
        self.scheduler: DecisionScheduler | None = None
        self.pending: list[PendingLabel] = []
        self.started_at = datetime.now(timezone.utc)
        self.clob_events = 0
        self.meaningful_clob_updates = 0
        self.last_clob_update_monotonic = 0.0
        self.latencies: list[float] = []
        self.labels_by_horizon = {horizon: 0 for horizon in HORIZONS}
        self.liquidity_warning = False
        self.resolution_pending = True

    async def run(self) -> None:
        if not os.environ.get("OPENROUTER_API_KEY", "").strip():
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        next_window = (int(time.time()) // 900 + 1) * 900
        print(f"Waiting for full {self.asset} market at {next_window} UTC epoch")
        rtds_task = asyncio.create_task(self._rtds_reader())
        try:
            await self._wait_for_boundary(next_window)
            opening = self.chainlink.capture_opening(next_window * 1000)
            if opening is None:
                raise RuntimeError("no Chainlink opening reference observed")
            market, session, config = await asyncio.to_thread(
                discover_forecast_market, self.asset, next_window
            )
            self.market, self.http_session, self.config = market, session, config
            self.clob = LiveMarketState(market.up_token, market.down_token)
            self.clob.tick_size = market.minimum_tick_size
            self.log = JsonlWriter(
                Path(__file__).parent
                / "data"
                / f"forecast_{self.asset.lower()}_{next_window}.jsonl"
            )
            self.log.write(
                "session_start",
                started_at=self.started_at.isoformat(),
                market_slug=market.slug,
                asset=self.asset,
                window_start=market.window_start,
                window_end=market.window_end,
                max_jev_cost=str(self.cost.maximum_usd),
                fee_schedule={
                    "enabled": market.fee_schedule.enabled,
                    "rate": str(market.fee_schedule.rate),
                    "exponent": str(market.fee_schedule.exponent),
                    "taker_only": market.fee_schedule.taker_only,
                    "rebate_rate": str(market.fee_schedule.rebate_rate),
                },
            )
            self.log.write(
                "chainlink_open",
                opening_chainlink_price_observed=str(opening.price),
                opening_chainlink_source_timestamp=opening.source_timestamp_ms,
                opening_offset_ms=opening.offset_ms,
                opening_reference_quality=opening.quality,
            )
            print(
                f"Opening Chainlink: {opening.price} | offset {opening.offset_ms}ms "
                f"| quality {opening.quality}"
            )
            self.scheduler = DecisionScheduler(
                self._decide,
                self._on_forecast,
                self._on_jev_error,
                self._can_start_decision,
            )
            clob_task = asyncio.create_task(self._clob_reader())
            label_task = asyncio.create_task(self._label_resolver())
            ticker_task = asyncio.create_task(self._second_ticker())
            await self._wait_for_market_end_and_resolution()
            if self.scheduler is not None:
                await self.scheduler.wait_idle()
            await self._resolve_all_due(force_market_end=True)
            self.stop.set()
            for task in (clob_task, label_task, ticker_task):
                task.cancel()
            await asyncio.gather(clob_task, label_task, ticker_task, return_exceptions=True)
            self._finish_session()
        finally:
            self.stop.set()
            rtds_task.cancel()
            await asyncio.gather(rtds_task, return_exceptions=True)
            self.client.__exit__(None, None, None)

    async def _wait_for_boundary(self, window_start: int) -> None:
        while time.time() < window_start:
            await asyncio.sleep(min(0.25, max(0.01, window_start - time.time())))
        deadline = window_start + 5
        while time.time() < deadline:
            if self.chainlink.raw_updates and (
                self.chainlink.raw_updates[-1].timestamp_ms >= window_start * 1000
            ):
                return
            self.state_changed.clear()
            try:
                await asyncio.wait_for(self.state_changed.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass

    async def _rtds_reader(self) -> None:
        while not self.stop.is_set():
            try:
                async with connect(RTDS_URL, ping_interval=None, close_timeout=2) as websocket:
                    await websocket.send(json.dumps(subscription_frame(RAW_TOPIC)))
                    await websocket.send(json.dumps(subscription_frame(TWAP60_TOPIC)))
                    print("RTDS connected")
                    heartbeat = asyncio.create_task(self._heartbeat(websocket, 5))
                    try:
                        async for raw in websocket:
                            if self.stop.is_set():
                                break
                            if raw == "PONG":
                                continue
                            try:
                                if not raw:
                                    continue
                                message = json.loads(raw)
                                if isinstance(message, Mapping) and self.chainlink.apply(message):
                                    self.state_changed.set()
                                    self._submit_latest()
                            except (ValueError, TypeError) as exc:
                                print(f"RTDS message ignored: {exc}", file=sys.stderr)
                    finally:
                        heartbeat.cancel()
                        await asyncio.gather(heartbeat, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self.stop.is_set():
                    print(f"RTDS disconnected: {exc}; reconnecting", file=sys.stderr)
                    await asyncio.sleep(1)

    async def _clob_reader(self) -> None:
        assert self.market is not None and self.clob is not None
        while not self.stop.is_set() and self.clob.resolved_winner is None:
            try:
                self.clob.reset_for_reconnect()
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
                    print("CLOB connected")
                    heartbeat = asyncio.create_task(self._heartbeat(websocket, 10))
                    shown_liquidity = False
                    try:
                        async for raw in websocket:
                            if self.stop.is_set():
                                break
                            if raw == "PONG":
                                continue
                            payload = json.loads(raw)
                            for message in iter_messages(payload):
                                self.clob_events += 1
                                try:
                                    meaningful = self.clob.apply_event(message)
                                except (ValueError, TypeError) as exc:
                                    print(f"CLOB message ignored: {exc}", file=sys.stderr)
                                    continue
                                if not meaningful:
                                    continue
                                self.meaningful_clob_updates += 1
                                self.last_clob_update_monotonic = time.monotonic()
                                if self.clob.trade_ready() and not shown_liquidity:
                                    self._print_liquidity()
                                    shown_liquidity = True
                                self._submit_latest()
                    finally:
                        heartbeat.cancel()
                        await asyncio.gather(heartbeat, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self.stop.is_set():
                    print(f"CLOB disconnected: {exc}; reconnecting", file=sys.stderr)
                    await asyncio.sleep(1)

    async def _heartbeat(self, websocket: Any, interval: int) -> None:
        while True:
            await asyncio.sleep(interval)
            await websocket.send("PING")

    async def _second_ticker(self) -> None:
        previous = int(time.time())
        while not self.stop.is_set():
            await asyncio.sleep(0.1)
            current = int(time.time())
            if current != previous:
                previous = current
                self._submit_latest()

    def _can_start_decision(self) -> bool:
        return bool(
            self.cost.can_request
            and self.market is not None
            and time.time() < self.market.window_end
            and self.clob is not None
            and self.clob.resolved_winner is None
        )

    def _submit_latest(self) -> None:
        if (
            self.scheduler is None
            or self.market is None
            or self.clob is None
            or not self.clob.trade_ready()
            or not self.chainlink.ready
            or not self._can_start_decision()
        ):
            return
        entries = hypothetical_entries(self.clob, self.market.fee_schedule)
        model_state = build_compact_state(
            self.market, self.chainlink, self.clob, entries
        )
        questions = build_questions(model_state["market"]["left_s"], entries)
        payload = {
            "jev_state": model_state,
            "questions": questions,
            "entries": entries,
            "captured_epoch": time.time(),
            "captured_monotonic": time.monotonic(),
            "liquidity": liquidity_snapshot(
                self.clob, self.market.fee_schedule, self.market.minimum_tick_size
            ),
            "chainlink": self.chainlink.logger_snapshot(),
        }
        self.scheduler.submit(payload)

    async def _decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await call_jev_nouls_async(
            self.client,
            JEV_MODEL,
            payload["jev_state"],
            payload["questions"],
        )

    async def _on_jev_error(self, exc: Exception) -> None:
        print(f"Jev error: {exc}", file=sys.stderr)

    async def _on_forecast(
        self,
        payload: dict[str, Any],
        result: dict[str, Any],
        started_at: datetime,
        finished_at: datetime,
    ) -> None:
        assert self.market is not None and self.log is not None
        decision_id = str(uuid.uuid4())
        answers = result["probabilities"]
        p_up = answers["resolve_up"]
        entries: dict[str, ExecutionQuote] = payload["entries"]
        edges = terminal_edges(p_up, entries)
        request_cost = self.cost.record(result.get("input_tokens"))
        self.latencies.append(float(result["latency_ms"]))
        entries_json = {
            side: quote.as_dict() for side, quote in entries.items()
            if quote.fully_executable
        }
        self.log.write(
            "forecast",
            decision_id=decision_id,
            decision_started_at=started_at.isoformat(),
            decision_finished_at=finished_at.isoformat(),
            market_slug=self.market.slug,
            time_remaining=payload["jev_state"]["market"]["left_s"],
            compact_state_sent_to_jev=payload["jev_state"],
            clob_liquidity_summary=payload["liquidity"],
            chainlink_state=payload["chainlink"],
            question_ids=list(payload["questions"]),
            answers=answers,
            p_resolve_up=p_up,
            p_resolve_down=1 - p_up,
            up_terminal_break_even=_decimal_or_none(edges["up_terminal_break_even"]),
            down_terminal_break_even=_decimal_or_none(edges["down_terminal_break_even"]),
            model_terminal_edge_up=_decimal_or_none(edges["model_terminal_edge_up"]),
            model_terminal_edge_down=_decimal_or_none(edges["model_terminal_edge_down"]),
            hypothetical_entries=entries_json,
            latency_ms=result["latency_ms"],
            input_tokens=result.get("input_tokens"),
            estimated_request_cost=str(request_cost),
        )
        for question_id, probability in answers.items():
            parsed = _parse_profit_question(question_id)
            if parsed is None:
                continue
            side, horizon = parsed
            entry = entries[side]
            self.pending.append(
                PendingLabel(
                    decision_id=decision_id,
                    side=side,
                    horizon=horizon,
                    probability=probability,
                    entry_epoch=payload["captured_epoch"],
                    target_epoch=payload["captured_epoch"] + horizon,
                    due_monotonic=payload["captured_monotonic"] + horizon,
                    entry_quote=entry,
                )
            )
        print(
            f"Forecast left={payload['jev_state']['market']['left_s']}s "
            f"pUP={p_up:.3f} questions={len(answers)} "
            f"tokens={result.get('input_tokens')} latency={result['latency_ms']:.0f}ms"
        )
        if not self.cost.can_request and not self.cost.exhausted_announced:
            self.cost.exhausted_announced = True
            print("Jev budget exhausted")

    async def _label_resolver(self) -> None:
        while not self.stop.is_set():
            await self._resolve_all_due()
            await asyncio.sleep(0.075)

    async def _resolve_all_due(self, force_market_end: bool = False) -> None:
        if self.market is None or self.clob is None or self.log is None:
            return
        now_mono = time.monotonic()
        due = [
            label
            for label in self.pending
            if force_market_end or label.due_monotonic <= now_mono
        ]
        if not due:
            return
        self.pending = [label for label in self.pending if label not in due]
        books = self.clob.execution_books()
        now_epoch = time.time()
        for label in due:
            effective_end = 0 if force_market_end else self.market.window_end
            result = evaluate_label(
                label,
                books[label.side]["bids"],
                self.market.fee_schedule,
                now_epoch,
                now_mono,
                self.last_clob_update_monotonic,
                effective_end,
            )
            self.labels_by_horizon[label.horizon] += int(result.status == "valid")
            self.log.write(
                "label",
                decision_id=label.decision_id,
                side=label.side,
                horizon=label.horizon,
                predicted_probability=label.probability,
                entry_timestamp=label.entry_epoch,
                due_timestamp=label.target_epoch,
                label_timestamp=now_epoch,
                entry_quote=label.entry_quote.as_dict(),
                entry_fee=str(label.entry_quote.fee),
                shares=str(label.entry_quote.shares),
                exit_quote=(
                    result.exit_quote.as_dict() if result.exit_quote is not None else None
                ),
                exit_fee=(
                    str(result.exit_quote.fee) if result.exit_quote is not None else None
                ),
                actual_net_pnl=_decimal_or_none(result.actual_net_pnl),
                profitable=result.profitable,
                label_status=result.status,
                delay_ms=result.delay_ms,
                book_age_ms=result.book_age_ms,
            )

    async def _wait_for_market_end_and_resolution(self) -> None:
        assert self.market is not None and self.clob is not None
        while time.time() < self.market.window_end:
            await asyncio.sleep(0.2)
        deadline = self.market.window_end + RESOLUTION_WAIT_SEC
        next_poll = self.market.window_end
        while time.time() < deadline and self.clob.resolved_winner is None:
            now = time.time()
            if now >= next_poll:
                winner = await asyncio.to_thread(self._poll_resolution)
                if winner is not None:
                    self.clob.resolved_winner = winner
                    break
                next_poll = now + RESOLUTION_POLL_SEC
            await asyncio.sleep(0.2)
        if self.clob.resolved_winner is not None:
            self.resolution_pending = False
            assert self.log is not None
            self.log.write(
                "market_resolved",
                resolved_at=datetime.now(timezone.utc).isoformat(),
                winner=self.clob.resolved_winner,
            )

    def _poll_resolution(self) -> str | None:
        assert self.market is not None and self.config is not None
        fresh = fetch_market(
            self.http_session,
            str(self.config["gamma_base_url"]),
            self.asset,
            self.market.window_start,
            float(self.config["http_timeout_sec"]),
            False,
            time.time(),
        )
        market = fresh["market"]
        if market.get("closed") is not True and fresh["event"].get("closed") is not True:
            return None
        return parse_resolved_outcome(market.get("outcomes"), market.get("outcomePrices"))

    def _print_liquidity(self) -> None:
        assert self.market is not None and self.clob is not None
        snapshot = liquidity_snapshot(
            self.clob, self.market.fee_schedule, self.market.minimum_tick_size
        )
        print("\nLiquidity snapshot:")
        for key, label in (("up", "UP"), ("down", "DOWN")):
            item = snapshot[key]
            print(f"{label} spread ticks: {item['spread_ticks']}")
            print(f"$10 {label} entry executable: {'yes' if item['can_buy_10'] else 'no'}")
            print(
                f"$10 immediate {label} roundtrip executable: "
                f"{'yes' if item['immediate_roundtrip_executable'] else 'no'}"
            )
        self.liquidity_warning = any(
            Decimal(snapshot[key]["spread_ticks"]) > 2
            or not snapshot[key]["can_buy_10"]
            for key in ("up", "down")
        )
        if self.liquidity_warning:
            print("WARNING: thin liquidity; this session may be noisier")

    def _finish_session(self) -> None:
        assert self.market is not None and self.clob is not None and self.log is not None
        metrics = self.cost.metrics()
        summary = {
            "market": self.market.slug,
            "winner": self.clob.resolved_winner,
            "settlement_pending": self.resolution_pending,
            "clob_events": self.clob_events,
            "chainlink_raw_events": self.chainlink.raw_events,
            "chainlink_twap_events": self.chainlink.twap_events,
            "jev_requests": len(self.cost.input_tokens),
            "avg_latency_ms": (
                sum(self.latencies) / len(self.latencies) if self.latencies else 0
            ),
            **metrics,
            "valid_labels": self.labels_by_horizon,
            "liquidity_warning": self.liquidity_warning,
        }
        self.log.write("session_summary", **summary)
        print("\nSession summary")
        for key, value in summary.items():
            print(f"{key}: {value}")
        from forecast_report import generate_report

        print("\n" + generate_report(self.log.path, resolve_missing=False))


def _parse_profit_question(question_id: str) -> tuple[str, int] | None:
    parts = question_id.split("_")
    if len(parts) != 3 or parts[1] != "profit" or not parts[2].endswith("s"):
        return None
    side = parts[0].upper()
    try:
        horizon = int(parts[2][:-1])
    except ValueError:
        return None
    if side not in ("UP", "DOWN") or horizon not in HORIZONS:
        return None
    return side, horizon


def _decimal_or_none(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime Jev forecast experiment")
    parser.add_argument("--asset", required=True, choices=("BTC", "ETH"))
    parser.add_argument("--max-jev-cost", type=Decimal, default=Decimal("0.15"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        asyncio.run(ForecastExperiment(args.asset, args.max_jev_cost).run())
        return 0
    except KeyboardInterrupt:
        print("forecast experiment stopped")
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
