"""Input-time V3 forecast features; no order execution or resolution logic."""

from __future__ import annotations

import json
import math
import statistics
from collections import deque
from decimal import Decimal
from typing import Any, Mapping

from chainlink_live import ChainlinkUpdate
from live_market import LiveMarketState
from paper_trader import Level


def trades_subscription() -> dict[str, Any]:
    # https://github.com/Polymarket/real-time-data-client#messages-hierarchy
    # Server-side slug filters currently suppress even active-market trades.
    # Subscribe to the public stream and require BOTH slug and token in apply().
    return {"action": "subscribe", "subscriptions": [{
        "topic": "activity", "type": "trades",
    }]}


class TradeFlow:
    """Rolling BUY notional by outcome. SELLs are not relabelled as BUYs."""

    def __init__(self, market_slug: str, up_token: str, down_token: str):
        self.market_slug = market_slug
        self.tokens = {up_token: "UP", down_token: "DOWN"}
        self.trades: deque[tuple[float, str, float, tuple[Any, ...]]] = deque()
        self.connected_since: float | None = None
        self.events = 0

    def prune(self, now: float) -> None:
        # RTDS delivery need not be ordered by exchange timestamp.
        self.trades = deque(item for item in self.trades if now - 60 <= item[0] <= now)

    def apply(self, message: Mapping[str, Any], now: float) -> bool:
        self.prune(now)
        if message.get("topic") != "activity" or message.get("type") != "trades":
            return False
        trade = message.get("payload")
        if not isinstance(trade, Mapping):
            return False
        token = str(trade.get("asset", ""))
        if token not in self.tokens or trade.get("slug") != self.market_slug:
            return False
        if str(trade.get("side", "")).upper() != "BUY":
            return False
        try:
            timestamp = float(trade["timestamp"])
            if timestamp >= 100_000_000_000:
                timestamp /= 1000
            price, size = float(trade["price"]), float(trade["size"])
        except (KeyError, TypeError, ValueError):
            return False
        if not all(math.isfinite(v) for v in (timestamp, price, size, price * size)):
            return False
        if not now - 60 <= timestamp <= now or not 0 < price < 1 or size <= 0:
            return False
        identity = (
            trade.get("transactionHash"), trade.get("proxyWallet"), token,
            timestamp, price, size,
        )
        if trade.get("transactionHash") and any(t[3] == identity for t in self.trades):
            return False
        self.trades.append((timestamp, self.tokens[token], price * size, identity))
        self.events += 1
        return True

    def snapshot(self, now: float) -> dict[str, Any]:
        self.prune(now)
        up = [t[2] for t in self.trades if t[1] == "UP"]
        down = [t[2] for t in self.trades if t[1] == "DOWN"]
        coverage = 0 if self.connected_since is None else min(60, max(0, now - self.connected_since))
        return {
            "up_count": len(up), "down_count": len(down),
            "avg_size": statistics.fmean(up + down) if up or down else 0.0,
            "up_avg_size": statistics.fmean(up) if up else 0.0,
            "down_avg_size": statistics.fmean(down) if down else 0.0,
            "up_usd": sum(up), "down_usd": sum(down),
            "net": len(up) - len(down), "imbalance_usd": sum(up) - sum(down),
            "coverage_sec": round(coverage, 1),
            "status": "unavailable" if self.connected_since is None else (
                "ready" if coverage >= 60 else "warming_up"
            ),
        }


class RawVolatility:
    """Time-bounded raw history, independent of the legacy 100-tick buffer."""

    def __init__(self):
        self.updates: deque[ChainlinkUpdate] = deque()

    def add(self, update: ChainlinkUpdate, now: float) -> None:
        if update.kind == "raw" and update.value > 0 and update.value.is_finite():
            self.updates.append(update)
        self._prune(now)

    def _prune(self, now: float) -> None:
        self.updates = deque(u for u in self.updates if (now - 60) * 1000 <= u.timestamp_ms <= now * 1000)

    def snapshot(self, now: float) -> dict[str, Any]:
        self._prune(now)
        # Reconnect replay and out-of-order ticks must not add artificial reversals.
        points = sorted({u.timestamp_ms: u.value for u in self.updates}.items())
        returns = [math.log(float(b[1] / a[1])) for a, b in zip(points, points[1:])]
        directions = [1 if r > 0 else -1 for r in returns if r != 0]
        changes = sum(a != b for a, b in zip(directions, directions[1:]))
        if len(returns) < 2:
            regime = "insufficient_data"
        elif not directions:
            regime = "flat"
        else:
            regime = "choppy" if changes >= 2 and changes / max(1, len(directions) - 1) >= 0.5 else "trend"
        return {
            "vol_60s": statistics.pstdev(returns) * 100 if len(returns) >= 2 else None,
            "vol_regime": regime, "vol_direction_changes": changes,
            "vol_samples": len(points),
            "vol_coverage_sec": (points[-1][0] - points[0][0]) / 1000 if points else 0.0,
        }


def detect_wall(levels: list[Level], total_depth_usd: Decimal, outcome: str, side: str) -> str | None:
    if len(levels) < 6:
        return None
    top = levels[0]
    average = sum((level.size for level in levels[1:6]), Decimal(0)) / 5
    notional = top.price * top.size
    if average <= 0 or top.size <= average * 3 or notional <= max(Decimal(1500), total_depth_usd * Decimal("0.25")):
        return None
    return f"{outcome} {side} wall {top.price} ${notional:.0f} ({top.size / average:.1f}x)"


def book_features(clob: LiveMarketState) -> dict[str, Any]:
    books = clob.execution_books()
    result: dict[str, Any] = {}
    walls: list[str] = []
    for side, outcome in (("UP", clob.up), ("DOWN", clob.down)):
        book = books[side]
        total = sum((level.price * level.size for levels in book.values() for level in levels), Decimal(0))
        top5 = sum((level.price * level.size for levels in book.values() for level in levels[:5]), Decimal(0))
        bid, ask = outcome.best_bid, outcome.best_ask
        assert bid is not None and ask is not None
        mid = (bid + ask) / 2
        result[side.lower()] = {
            "bid": float(bid), "ask": float(ask), "mid": float(mid),
            "bid_usd": float(bid * outcome.bids.get(bid, Decimal(0))),
            "ask_usd": float(ask * outcome.asks.get(ask, Decimal(0))),
            "top5_depth_usd": float(top5), "total_depth_usd": float(total),
            "spread_bps": float((ask - bid) / mid * 10000) if mid > 0 else None,
        }
        for key, label in (("bids", "BID"), ("asks", "ASK")):
            # Only a level at the current best quote can be a top wall.
            levels = book[key]
            if levels and levels[0].price == (bid if key == "bids" else ask):
                wall = detect_wall(levels, total, side, label)
                if wall:
                    walls.append(wall)
    return {
        "book": result, "mid": result["up"]["mid"],
        "spread_bps": result["up"]["spread_bps"],
        "depth_ratio": result["up"]["top5_depth_usd"] / result["down"]["top5_depth_usd"] if result["down"]["top5_depth_usd"] else None,
        "wall_text": "; ".join(walls) if walls else "none detected",
    }


def gate_reason(mid: float) -> str | None:
    if mid < 0.10:
        return "mid_below_0.10"
    if mid > 0.90:
        return "mid_above_0.90"
    return None


def render_meta_state(asset: str, rule: str, chain: Mapping[str, Any], features: Mapping[str, Any], economics: Mapping[str, Any]) -> str:
    left = features["time_left_sec"]
    flow = features["flow_60s"]
    up, down = features["book"]["up"], features["book"]["down"]
    spread = features["spread_bps"]
    ratio = features["depth_ratio"]
    vol = features["vol_60s"]
    lines = [
        f"{asset} 15-minute UP/DOWN contract. Resolution rule: {rule}",
        f"Time left: {left}s ({(900 - left) / 9:.0f}% elapsed)",
        f"Chainlink start {chain['open']}, raw {chain['raw']}, distance from start {features['distance_from_start_bps']:+.2f} bps; TWAP60 {chain['twap60']} ({chain['twap_from_open_bps']:+.2f} bps from start).",
        f"Raw changes: last 1/5/20 updates {chain['raw_d1']}/{chain['raw_d5']}/{chain['raw_d20']} bps.",
        f"Fees: {features['fees_bps'] / 100:g}% taker schedule rate; exponent {features['fees_exponent']:g}. Effective $10 BUY fees: UP {features['effective_buy_fees_bps']['UP']} bps, DOWN {features['effective_buy_fees_bps']['DOWN']} bps.",
        f"Flow 60s: {flow['up_count']} BUY UP (avg ${flow['up_avg_size']:.0f}) vs {flow['down_count']} BUY DOWN (avg ${flow['down_avg_size']:.0f}), net {flow['net']:+d} UP (${flow['imbalance_usd']:+.0f}); {flow['status']}, coverage {flow['coverage_sec']:.0f}s.",
        f"Book UP bid {up['bid']:.3f} ${up['bid_usd']:.0f} / ask {up['ask']:.3f} ${up['ask_usd']:.0f}, DOWN {down['bid']:.3f} ${down['bid_usd']:.0f} / {down['ask']:.3f} ${down['ask_usd']:.0f}, spread {spread / 100:.2f}% ({spread:.1f} bps), depth UP/DOWN " + (f"{ratio:.2f}x" if ratio is not None else "unavailable") + " (top 5 bids + asks, USD).",
        f"Wall: {features['wall_text']}",
        "Chainlink vol 60s: " + (f"{vol:.4f}%" if vol is not None else "unavailable") + f" {features['vol_regime']}; {features['vol_direction_changes']} direction changes, {features['vol_samples']} samples spanning {features['vol_coverage_sec']:.0f}s.",
        "Terminal break-even after entry fees: " + json.dumps(economics, separators=(",", ":")),
        "Q1 is the final UP resolution probability. Assess distance together with time left, raw volatility, flow and book liquidity. Current raw/TWAP side is an observation, not a certain final outcome. Fees affect execution economics, not the resolution rule. Missing or partial flow/volatility is not evidence of no activity. Output one calibrated probability.",
    ]
    return "\n".join(lines)
