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
from forecast_features import (
    RawVolatility, TradeFlow, book_features, gate_reason,
    render_meta_state, trades_subscription,
)
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


@dataclass(frozen=True)
class ForecastPendingLabel:
    decision_id: str
    horizon: int
    target_epoch: float
    due_monotonic: float
    response_p_market_up: float
    p_jev_up: float
    response_entries: Mapping[str, ExecutionQuote]


@dataclass(frozen=True)
class SideFutureResult:
    status: str
    exit_quote: ExecutionQuote | None
    net_pnl: Decimal | None


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
    time_remaining: int | None = None,
    entries: Mapping[str, ExecutionQuote] | None = None,
) -> dict[str, str]:
    return {"p_final_up": RESOLVE_INSTRUCTION}


def normalized_market_probabilities(
    up_bid: Decimal,
    up_ask: Decimal,
    down_bid: Decimal,
    down_ask: Decimal,
) -> dict[str, Decimal]:
    up_mid = (up_bid + up_ask) / Decimal("2")
    down_mid = (down_bid + down_ask) / Decimal("2")
    denominator = up_mid + down_mid
    if denominator <= 0:
        raise ValueError("market midpoint denominator must be positive")
    return {
        "up_mid": up_mid,
        "down_mid": down_mid,
        "mid_sum": denominator,
        "p_market_up": up_mid / denominator,
        "p_market_down": down_mid / denominator,
    }


def market_probabilities(clob: LiveMarketState) -> dict[str, Decimal]:
    if not clob.trade_ready():
        raise RuntimeError("CLOB state is not trade-ready")
    return normalized_market_probabilities(
        clob.up.best_bid,
        clob.up.best_ask,
        clob.down.best_bid,
        clob.down.best_ask,
    )


def probability_sides(
    p_jev_up: float, p_market_up: Decimal, epsilon: Decimal = Decimal("0.000000001")
) -> dict[str, Any]:
    jev_up = Decimal(str(p_jev_up))
    jev_down = Decimal("1") - jev_up
    market_down = Decimal("1") - p_market_up
    delta_up = jev_up - p_market_up
    delta_down = jev_down - market_down
    relative = "NONE"
    if delta_up > epsilon:
        relative = "UP"
    elif delta_down > epsilon:
        relative = "DOWN"
    return {
        "p_jev_up": jev_up,
        "p_jev_down": jev_down,
        "probability_delta_up": delta_up,
        "probability_delta_down": delta_down,
        "relative_side": relative,
        "jev_side": "UP" if jev_up >= Decimal("0.5") else "DOWN",
        "market_side": "UP" if p_market_up >= Decimal("0.5") else "DOWN",
    }


def chainlink_sides(chainlink: ChainlinkLiveState) -> dict[str, str]:
    if not chainlink.ready or chainlink.opening is None:
        raise RuntimeError("Chainlink state is not ready")
    opening = chainlink.opening.price
    return {
        "raw_side": "UP" if chainlink.raw_updates[-1].value >= opening else "DOWN",
        "twap_side": "UP" if chainlink.twap_updates[-1].value >= opening else "DOWN",
    }


def select_best_edge(edges: Mapping[str, Decimal | None]) -> dict[str, Any]:
    up = edges.get("model_terminal_edge_up")
    down = edges.get("model_terminal_edge_down")
    up_value = up if up is not None else Decimal("-Infinity")
    down_value = down if down is not None else Decimal("-Infinity")
    if up_value > down_value and up_value > 0:
        side = "UP"
        best = up
        opposite = down
    elif down_value > up_value and down_value > 0:
        side = "DOWN"
        best = down
        opposite = up
    else:
        side = "NONE"
        best = max(up_value, down_value)
        if not best.is_finite():
            best = None
        opposite = min(up_value, down_value)
        if not opposite.is_finite():
            opposite = None
    return {
        "best_edge_side": side,
        "best_edge_value": best,
        "opposite_edge_value": opposite,
    }


def signed_repricing(
    p_jev_up: float, response_p_market_up: float, future_p_market_up: float
) -> float:
    delta = p_jev_up - response_p_market_up
    direction = 1 if delta > 0 else -1 if delta < 0 else 0
    return direction * (future_p_market_up - response_p_market_up)


def disagreement_side_pnl(
    p_jev_up: float,
    p_market_up: float,
    up_pnl: Decimal | None,
    down_pnl: Decimal | None,
) -> Decimal | None:
    if p_jev_up > p_market_up:
        return up_pnl
    if p_jev_up < p_market_up:
        return down_pnl
    return None


def classify_winner_loser_edges(
    winner: str, edge_up: Decimal | None, edge_down: Decimal | None
) -> str:
    winner_edge = edge_up if winner == "UP" else edge_down
    loser_edge = edge_down if winner == "UP" else edge_up
    winner_positive = winner_edge is not None and winner_edge > 0
    loser_positive = loser_edge is not None and loser_edge > 0
    if winner_positive and loser_positive:
        return "BOTH"
    if winner_positive:
        return "WINNER_EDGE_ONLY"
    if loser_positive:
        return "LOSER_EDGE_ONLY"
    return "NEITHER"


def detect_transitions(
    values: list[tuple[float, str, int]], source: str
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    previous: str | None = None
    for timestamp, side, time_remaining in values:
        if previous is not None and side != previous:
            transitions.append(
                {
                    "source": source,
                    "timestamp": timestamp,
                    "from_side": previous,
                    "to_side": side,
                    "time_remaining": time_remaining,
                }
            )
        previous = side
    return transitions


def winner_relative_metrics(record: Mapping[str, Any], winner: str) -> dict[str, float]:
    p_jev_up = float(record["p_jev_up"])
    p_market_up = float(record["response_market_probabilities"]["p_market_up"])
    y_up = 1.0 if winner == "UP" else 0.0
    return {
        "p_jev_winner": p_jev_up if winner == "UP" else 1 - p_jev_up,
        "p_market_winner": p_market_up if winner == "UP" else 1 - p_market_up,
        "jev_brier": (p_jev_up - y_up) ** 2,
        "market_brier": (p_market_up - y_up) ** 2,
    }


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
    chainlink_compact = chainlink.compact()
    chainlink_compact.pop("twap_recent", None)
    state: dict[str, Any] = {
        "market": {
            "asset": market.asset,
            "left_s": max(0, int(market.window_end - current_time)),
            "rule": market.rules,
        },
        "chainlink": chainlink_compact,
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
            "terminal_break_even": _model_decimal(
                terminal_break_even(entry), Decimal("0.00001")
            ),
        }
    state["terminal_economics"] = economics
    return state


def build_enriched_state(
    market: MarketInfo,
    chainlink: ChainlinkLiveState,
    clob: LiveMarketState,
    entries: Mapping[str, ExecutionQuote],
    flow: TradeFlow,
    volatility: RawVolatility,
    now: float,
) -> tuple[dict[str, str], dict[str, Any]]:
    chain = chainlink.compact()
    schedule = market.fee_schedule
    features = {
        "time_left_sec": max(0, int(market.window_start + 900 - now)),
        "distance_from_start_bps": chain["raw_from_open_bps"],
        "flow_60s": flow.snapshot(now),
        **book_features(clob),
        **volatility.snapshot(now),
        "fees_bps": float(schedule.rate * 10000) if schedule.enabled else 0.0,
        "fees_exponent": float(schedule.exponent),
        "effective_buy_fees_bps": {
            side: float(entry.fee / entry.gross_value * 10000)
            if entry.fully_executable and entry.gross_value > 0 else None
            for side, entry in entries.items()
        },
    }
    economics = {
        side: str(terminal_break_even(entry)) if entry.fully_executable else "unexecutable"
        for side, entry in entries.items()
    }
    text = render_meta_state(market.asset, market.rules, chain, features, economics)
    # The Decisions SDK expects an object; its Meta content is plain text.
    return {"meta": text}, features


def capture_realtime_state(
    market: MarketInfo,
    chainlink: ChainlinkLiveState,
    clob: LiveMarketState,
    entries: Mapping[str, ExecutionQuote],
    timestamp: float,
) -> dict[str, Any]:
    probabilities = market_probabilities(clob)
    chain = chainlink.compact()
    result: dict[str, Any] = {
        "timestamp": timestamp,
        "time_remaining": max(0, int(market.window_end - timestamp)),
        "chainlink": {
            "raw": chain["raw"],
            "twap60": chain["twap60"],
            "raw_from_open_bps": chain["raw_from_open_bps"],
            "twap_from_open_bps": chain["twap_from_open_bps"],
        },
        "market_probabilities": {
            key: str(value) for key, value in probabilities.items()
        },
        "entries": {
            side: _entry_dict(quote)
            for side, quote in entries.items()
        },
    }
    books = clob.execution_books()
    for side, key in (("UP", "up"), ("DOWN", "down")):
        outcome = clob.up if side == "UP" else clob.down
        bid_depth = sum((level.size for level in books[side]["bids"][:5]), Decimal("0"))
        ask_depth = sum((level.size for level in books[side]["asks"][:5]), Decimal("0"))
        result[key] = {
            "bid": str(outcome.best_bid),
            "ask": str(outcome.best_ask),
            "mid": str(probabilities[f"{key}_mid"]),
            "depth_bid": str(bid_depth),
            "depth_ask": str(ask_depth),
        }
    return result


def evaluate_future_side(
    entry: ExecutionQuote,
    bids: list[Level],
    schedule: FeeSchedule,
    base_status: str,
) -> SideFutureResult:
    if base_status != "valid":
        return SideFutureResult(base_status, None, None)
    if not entry.fully_executable:
        return SideFutureResult("unexecutable", None, None)
    exit_quote = quote_sell(bids, entry.shares, schedule)
    if not exit_quote.fully_executable:
        return SideFutureResult("unexecutable", exit_quote, None)
    return SideFutureResult(
        "valid", exit_quote, exit_quote.net_cash - entry.net_cash
    )


def create_pending_labels(
    decision_id: str,
    response_epoch: float,
    response_monotonic: float,
    response_p_market_up: float,
    p_jev_up: float,
    response_entries: Mapping[str, ExecutionQuote],
) -> list[ForecastPendingLabel]:
    return [
        ForecastPendingLabel(
            decision_id=decision_id,
            horizon=horizon,
            target_epoch=response_epoch + horizon,
            due_monotonic=response_monotonic + horizon,
            response_p_market_up=response_p_market_up,
            p_jev_up=p_jev_up,
            response_entries=response_entries,
        )
        for horizon in HORIZONS
    ]


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
        self.volatility = RawVolatility()
        self.flow: TradeFlow | None = None
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
        self.pending: list[ForecastPendingLabel] = []
        self.last_signs: dict[str, str] = {}
        self.started_at = datetime.now(timezone.utc)
        self.clob_events = 0
        self.meaningful_clob_updates = 0
        self.last_clob_update_monotonic = 0.0
        self.latencies: list[float] = []
        self.labels_by_horizon = {horizon: 0 for horizon in HORIZONS}
        self.liquidity_warning = False
        self.resolution_pending = True
        self.last_sample_second: int | None = None
        self.skipped = 0

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
            self.flow = TradeFlow(market.slug, market.up_token, market.down_token)
            self.clob = LiveMarketState(market.up_token, market.down_token)
            self.clob.tick_size = market.minimum_tick_size
            self.log = JsonlWriter(
                Path(__file__).parent
                / "data"
                / f"forecast_{self.asset.lower()}_{next_window}.jsonl"
            )
            self.log.write(
                "session_start",
                experiment_version=3,
                schema_version=3,
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
            trades_task = asyncio.create_task(self._trades_reader())
            label_task = asyncio.create_task(self._label_resolver())
            ticker_task = asyncio.create_task(self._second_ticker())
            await self._wait_for_market_end_and_resolution()
            if self.scheduler is not None:
                await self.scheduler.wait_idle()
            await self._resolve_all_due(force_market_end=True)
            self.stop.set()
            for task in (clob_task, trades_task, label_task, ticker_task):
                task.cancel()
            await asyncio.gather(clob_task, trades_task, label_task, ticker_task, return_exceptions=True)
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
                                    if message.get("topic") == RAW_TOPIC:
                                        self.volatility.add(self.chainlink.raw_updates[-1], time.time())
                                    self.state_changed.set()
                                    self._record_observed_crossovers()
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

    async def _trades_reader(self) -> None:
        assert self.market is not None and self.flow is not None
        while not self.stop.is_set():
            try:
                async with connect(RTDS_URL, ping_interval=None, close_timeout=2) as websocket:
                    await websocket.send(json.dumps(trades_subscription()))
                    self.flow.connected_since = time.time()
                    print(f"RTDS trades connected: {self.market.slug}")
                    heartbeat = asyncio.create_task(self._heartbeat(websocket, 5))
                    try:
                        async for raw in websocket:
                            if self.stop.is_set():
                                break
                            if not raw or raw == "PONG":
                                continue
                            try:
                                for message in iter_messages(json.loads(raw)):
                                    if self.flow.apply(message, time.time()):
                                        self._submit_latest()
                            except (ValueError, TypeError) as exc:
                                print(f"RTDS trade ignored: {exc}", file=sys.stderr)
                    finally:
                        self.flow.connected_since = None
                        heartbeat.cancel()
                        await asyncio.gather(heartbeat, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self.stop.is_set():
                    print(f"RTDS trades disconnected: {exc}; reconnecting", file=sys.stderr)
                    await asyncio.sleep(1)
            finally:
                self.flow.connected_since = None

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
                                self._record_observed_crossovers()
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
            self.market is not None
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
        # One candidate per second, coalesced during inference. This also keeps
        # gate logging bounded when no API latency limits the scheduler.
        sample_second = int(time.time())
        if sample_second == self.last_sample_second:
            return
        self.last_sample_second = sample_second
        payload = {
            "jev_state": {"sample_second": sample_second},
            "questions": build_questions(),
        }
        self.scheduler.submit(payload)

    async def _decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.market is not None and self.clob is not None and self.flow is not None
        if not self.clob.trade_ready() or not self.chainlink.ready:
            raise RuntimeError("input state is not ready")
        input_epoch = time.time()
        input_entries = hypothetical_entries(self.clob, self.market.fee_schedule)
        payload["input_timestamp"] = input_epoch
        payload["input_entries"] = input_entries
        payload["input_state"] = capture_realtime_state(
            self.market, self.chainlink, self.clob, input_entries, input_epoch
        )
        payload["jev_state"], payload["features"] = build_enriched_state(
            self.market, self.chainlink, self.clob, input_entries,
            self.flow, self.volatility, input_epoch,
        )
        reason = gate_reason(payload["features"]["mid"])
        if reason is None and not self.cost.can_request:
            reason = "budget_exhausted"
        if reason is not None:
            return {"skipped": True, "skip_reason": reason}
        print(payload["jev_state"]["meta"])
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
        assert self.market is not None and self.log is not None and self.clob is not None
        decision_id = str(uuid.uuid4())
        if result.get("skipped"):
            self.skipped += 1
            self.log.write(
                "forecast_skip", experiment_version=3, schema_version=3,
                decision_id=decision_id, market_slug=self.market.slug,
                input_timestamp=payload["input_timestamp"],
                input_state=payload["input_state"],
                state_text=payload["jev_state"]["meta"],
                skipped=True, skip_reason=result["skip_reason"],
                **payload["features"],
            )
            print(f"SKIP {result['skip_reason']}\n{payload['jev_state']['meta']}")
            return
        answers = result["probabilities"]
        p_up = answers["p_final_up"]
        input_entries: dict[str, ExecutionQuote] = payload["input_entries"]
        input_edges = terminal_edges(p_up, input_entries)
        response_epoch = time.time()
        response_monotonic = time.monotonic()
        response_entries = hypothetical_entries(self.clob, self.market.fee_schedule)
        response_state = capture_realtime_state(
            self.market, self.chainlink, self.clob, response_entries, response_epoch
        )
        response_market = {
            key: Decimal(value)
            for key, value in response_state["market_probabilities"].items()
        }
        input_market = {
            key: Decimal(value)
            for key, value in payload["input_state"]["market_probabilities"].items()
        }
        sides = probability_sides(p_up, response_market["p_market_up"])
        baselines = chainlink_sides(self.chainlink)
        actionable_edges = terminal_edges(p_up, response_entries)
        best_edge = select_best_edge(actionable_edges)
        request_cost = self.cost.record(result.get("input_tokens"))
        self.latencies.append(float(result["latency_ms"]))
        self.log.write(
            "forecast",
            experiment_version=3,
            schema_version=3,
            decision_id=decision_id,
            input_timestamp=payload["input_timestamp"],
            response_timestamp=response_epoch,
            decision_started_at=started_at.isoformat(),
            decision_finished_at=finished_at.isoformat(),
            market_slug=self.market.slug,
            time_remaining=response_state["time_remaining"],
            input_state=payload["input_state"],
            response_state=response_state,
            compact_state_sent_to_jev=payload["jev_state"],
            state_text=payload["jev_state"]["meta"],
            **payload["features"],
            skipped=False,
            skip_reason=None,
            question_ids=["p_final_up"],
            p_final_up=p_up,
            p_jev_up=p_up,
            p_jev_down=1 - p_up,
            input_market_probabilities={
                key: str(value) for key, value in input_market.items()
            },
            response_market_probabilities={
                key: str(value) for key, value in response_market.items()
            },
            probability_delta_up=str(sides["probability_delta_up"]),
            probability_delta_down=str(sides["probability_delta_down"]),
            relative_side=sides["relative_side"],
            raw_side=baselines["raw_side"],
            twap_side=baselines["twap_side"],
            jev_side=sides["jev_side"],
            market_side=sides["market_side"],
            oracle_entries={side: _entry_dict(quote) for side, quote in input_entries.items()},
            response_entries={side: _entry_dict(quote) for side, quote in response_entries.items()},
            oracle_edge_up=_decimal_or_none(input_edges["model_terminal_edge_up"]),
            oracle_edge_down=_decimal_or_none(input_edges["model_terminal_edge_down"]),
            actionable_edge_up=_decimal_or_none(
                actionable_edges["model_terminal_edge_up"]
            ),
            actionable_edge_down=_decimal_or_none(
                actionable_edges["model_terminal_edge_down"]
            ),
            terminal_break_even_up=_decimal_or_none(
                actionable_edges["up_terminal_break_even"]
            ),
            terminal_break_even_down=_decimal_or_none(
                actionable_edges["down_terminal_break_even"]
            ),
            best_edge_side=best_edge["best_edge_side"],
            best_edge_value=_decimal_or_none(best_edge["best_edge_value"]),
            opposite_edge_value=_decimal_or_none(best_edge["opposite_edge_value"]),
            latency_ms=result["latency_ms"],
            input_tokens=result.get("input_tokens"),
            request_cost=str(request_cost),
        )
        self.pending.extend(
            create_pending_labels(
                decision_id,
                response_epoch,
                response_monotonic,
                float(response_market["p_market_up"]),
                p_up,
                response_entries,
            )
        )
        self._record_crossover("JEV", sides["jev_side"], response_epoch)
        print(
            f"Forecast left={response_state['time_remaining']}s "
            f"pUP={p_up:.3f} questions=1 "
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
            delay_ms = max(0.0, (now_epoch - label.target_epoch) * 1000)
            book_age_ms = (
                (now_mono - self.last_clob_update_monotonic) * 1000
                if self.last_clob_update_monotonic > 0
                else None
            )
            if force_market_end or now_epoch >= self.market.window_end:
                base_status = "market_ended"
            elif book_age_ms is None or book_age_ms > STALE_BOOK_SEC * 1000:
                base_status = "stale"
            elif not self.clob.trade_ready():
                base_status = "stale"
            else:
                base_status = "valid"
            up_result = evaluate_future_side(
                label.response_entries["UP"],
                books["UP"]["bids"],
                self.market.fee_schedule,
                base_status,
            )
            down_result = evaluate_future_side(
                label.response_entries["DOWN"],
                books["DOWN"]["bids"],
                self.market.fee_schedule,
                base_status,
            )
            future_state = None
            future_probabilities = None
            repricing = None
            if self.clob.trade_ready() and self.chainlink.ready:
                current_entries = hypothetical_entries(self.clob, self.market.fee_schedule)
                future_state = capture_realtime_state(
                    self.market, self.chainlink, self.clob, current_entries, now_epoch
                )
                future_probabilities = future_state["market_probabilities"]
            if base_status == "valid" and future_probabilities is not None:
                repricing = signed_repricing(
                    label.p_jev_up,
                    label.response_p_market_up,
                    float(future_probabilities["p_market_up"]),
                )
            self.labels_by_horizon[label.horizon] += int(base_status == "valid")
            self.log.write(
                "label",
                experiment_version=3,
                schema_version=3,
                decision_id=label.decision_id,
                horizon=label.horizon,
                target_timestamp=label.target_epoch,
                actual_timestamp=now_epoch,
                future_market_state=future_state,
                future_p_market_up=(
                    future_probabilities["p_market_up"] if future_probabilities else None
                ),
                future_p_market_down=(
                    future_probabilities["p_market_down"] if future_probabilities else None
                ),
                signed_repricing=repricing,
                up_response_entry_result=_side_future_dict(up_result),
                down_response_entry_result=_side_future_dict(down_result),
                label_status=base_status,
                book_age_ms=book_age_ms,
                delay_ms=delay_ms,
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

    def _record_observed_crossovers(self) -> None:
        if self.market is None or self.log is None:
            return
        timestamp = time.time()
        if self.chainlink.ready:
            baselines = chainlink_sides(self.chainlink)
            self._record_crossover("RAW", baselines["raw_side"], timestamp)
            self._record_crossover("TWAP", baselines["twap_side"], timestamp)
        if self.clob is not None and self.clob.trade_ready():
            market = market_probabilities(self.clob)
            side = "UP" if market["p_market_up"] >= Decimal("0.5") else "DOWN"
            self._record_crossover("MARKET", side, timestamp)

    def _record_crossover(self, source: str, side: str, timestamp: float) -> None:
        if self.market is None or self.log is None:
            return
        previous = self.last_signs.get(source)
        self.last_signs[source] = side
        if previous is None or previous == side:
            return
        self.log.write(
            "crossover",
            experiment_version=3,
            schema_version=3,
            source=source,
            timestamp=timestamp,
            from_side=previous,
            to_side=side,
            time_remaining=max(0, int(self.market.window_end - timestamp)),
        )

    def _finish_session(self) -> None:
        assert self.market is not None and self.clob is not None and self.log is not None
        metrics = self.cost.metrics()
        summary = {
            "experiment_version": 3,
            "schema_version": 3,
            "market": self.market.slug,
            "winner": self.clob.resolved_winner,
            "settlement_pending": self.resolution_pending,
            "clob_events": self.clob_events,
            "chainlink_raw_events": self.chainlink.raw_events,
            "chainlink_twap_events": self.chainlink.twap_events,
            "jev_requests": len(self.cost.input_tokens),
            "skipped": self.skipped,
            "trades_buy_events": self.flow.events if self.flow else 0,
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


def _decimal_or_none(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _entry_dict(quote: ExecutionQuote) -> dict[str, Any]:
    result = quote.as_dict()
    result["entry_status"] = "valid" if quote.fully_executable else "unexecutable"
    return result


def _side_future_dict(result: SideFutureResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "net_pnl": _decimal_or_none(result.net_pnl),
        "exit_quote": result.exit_quote.as_dict() if result.exit_quote else None,
    }


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
