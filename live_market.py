"""In-memory Polymarket order books for the public market WebSocket feed."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterable, Mapping

from paper_trader import Level


ZERO = Decimal("0")


def _value(data: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return None


def _decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"invalid decimal value: {value!r}") from exc


@dataclass
class OutcomeState:
    token_id: str
    bids: dict[Decimal, Decimal] = field(default_factory=dict)
    asks: dict[Decimal, Decimal] = field(default_factory=dict)
    reported_best_bid: Decimal | None = None
    reported_best_ask: Decimal | None = None
    last_trade_price: Decimal | None = None
    last_trade_size: Decimal | None = None
    last_trade_side: str | None = None
    recent_mids: deque[Decimal] = field(default_factory=lambda: deque(maxlen=50))
    recent_trades: deque[dict[str, str | None]] = field(
        default_factory=lambda: deque(maxlen=10)
    )
    last_update_timestamp: str | None = None

    @property
    def best_bid(self) -> Decimal | None:
        return self.reported_best_bid

    @property
    def best_ask(self) -> Decimal | None:
        return self.reported_best_ask

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.reported_best_bid = None
        self.reported_best_ask = None

    def apply_book(self, data: Mapping[str, Any]) -> None:
        self.bids = _levels_to_map(data.get("bids", []))
        self.asks = _levels_to_map(data.get("asks", []))
        self.reported_best_bid = max(self.bids, default=None)
        self.reported_best_ask = min(self.asks, default=None)
        last_trade = _decimal(_value(data, "last_trade_price", "lastTradePrice"))
        if last_trade is not None:
            self.last_trade_price = last_trade
        self.last_update_timestamp = str(data.get("timestamp") or "") or None
        self.record_mid()

    def apply_price_change(self, change: Mapping[str, Any]) -> None:
        price = _decimal(change.get("price"))
        size = _decimal(change.get("size"))
        side = str(change.get("side") or "").upper()
        if price is None or size is None or side not in ("BUY", "SELL"):
            raise ValueError("malformed price_change")
        levels = self.bids if side == "BUY" else self.asks
        if size == ZERO:
            levels.pop(price, None)
        else:
            levels[price] = size
        bid = _decimal(_value(change, "best_bid", "bestBid"))
        ask = _decimal(_value(change, "best_ask", "bestAsk"))
        self.reported_best_bid = bid if bid is not None else max(self.bids, default=None)
        self.reported_best_ask = ask if ask is not None else min(self.asks, default=None)
        self.record_mid()

    def apply_best_bid_ask(self, data: Mapping[str, Any]) -> None:
        self.reported_best_bid = _decimal(_value(data, "best_bid", "bestBid"))
        self.reported_best_ask = _decimal(_value(data, "best_ask", "bestAsk"))
        self.last_update_timestamp = str(data.get("timestamp") or "") or None
        self.record_mid()

    def apply_trade(self, data: Mapping[str, Any]) -> None:
        self.last_trade_price = _decimal(data.get("price"))
        self.last_trade_size = _decimal(data.get("size"))
        self.last_trade_side = str(data.get("side") or "").upper() or None
        self.last_update_timestamp = str(data.get("timestamp") or "") or None
        self.recent_trades.append(
            {
                "price": None if self.last_trade_price is None else str(self.last_trade_price),
                "size": None if self.last_trade_size is None else str(self.last_trade_size),
                "side": self.last_trade_side,
                "timestamp": self.last_update_timestamp,
            }
        )
        self.record_mid()

    def record_mid(self) -> None:
        if self.best_bid is not None and self.best_ask is not None:
            mid = (self.best_bid + self.best_ask) / Decimal("2")
            if not self.recent_mids or self.recent_mids[-1] != mid:
                self.recent_mids.append(mid)

    def top_levels(self, side: str, limit: int | None = None) -> list[Level]:
        source = self.bids if side == "bids" else self.asks
        reverse = side == "bids"
        levels = [Level(price, source[price]) for price in sorted(source, reverse=reverse)]
        return levels if limit is None else levels[:limit]

    def execution_ready(self) -> bool:
        if self.best_bid is None or self.best_ask is None:
            return False
        if not (ZERO < self.best_bid < Decimal("1") and ZERO < self.best_ask < Decimal("1")):
            return False
        bids = self.top_levels("bids", 1)
        asks = self.top_levels("asks", 1)
        return bool(
            bids
            and asks
            and bids[0].price == self.best_bid
            and asks[0].price == self.best_ask
        )

    def snapshot(self) -> dict[str, Any]:
        bids = self.top_levels("bids", 5)
        asks = self.top_levels("asks", 5)
        bid_size = sum((level.size for level in bids), ZERO)
        ask_size = sum((level.size for level in asks), ZERO)
        denominator = bid_size + ask_size
        imbalance = ZERO if denominator == ZERO else (bid_size - ask_size) / denominator
        return {
            "best_bid": str(self.best_bid),
            "best_ask": str(self.best_ask),
            "spread": str(self.best_ask - self.best_bid),
            "best_bid_size": str(bids[0].size) if bids else None,
            "best_ask_size": str(asks[0].size) if asks else None,
            "book_imbalance": str(imbalance),
            "top_bids": [_level_dict(level) for level in bids],
            "top_asks": [_level_dict(level) for level in asks],
            "last_trade": {
                "price": None if self.last_trade_price is None else str(self.last_trade_price),
                "size": None if self.last_trade_size is None else str(self.last_trade_size),
                "side": self.last_trade_side,
            },
            "mid_change_bps": {
                "last1": self._mid_change(1),
                "last5": self._mid_change(5),
                "last20": self._mid_change(20),
            },
            "recent_mids": [str(value) for value in list(self.recent_mids)[-10:]],
            "recent_trades": list(self.recent_trades),
        }

    def _mid_change(self, periods: int) -> float | None:
        if len(self.recent_mids) <= periods:
            return None
        old = self.recent_mids[-1 - periods]
        if old == ZERO:
            return None
        return float((self.recent_mids[-1] - old) / old * Decimal("10000"))


class LiveMarketState:
    def __init__(self, up_token: str, down_token: str):
        self.by_token = {
            up_token: OutcomeState(up_token),
            down_token: OutcomeState(down_token),
        }
        self.up_token = up_token
        self.down_token = down_token
        self.tick_size: Decimal | None = None
        self.resolved_winner: str | None = None

    @property
    def up(self) -> OutcomeState:
        return self.by_token[self.up_token]

    @property
    def down(self) -> OutcomeState:
        return self.by_token[self.down_token]

    def reset_for_reconnect(self) -> None:
        for outcome in self.by_token.values():
            outcome.reset()

    def trade_ready(self) -> bool:
        return self.resolved_winner is None and self.up.execution_ready() and self.down.execution_ready()

    def execution_books(self) -> dict[str, dict[str, list[Level]]]:
        return {
            "UP": {"bids": self.up.top_levels("bids"), "asks": self.up.top_levels("asks")},
            "DOWN": {"bids": self.down.top_levels("bids"), "asks": self.down.top_levels("asks")},
        }

    def summary(self) -> dict[str, Any]:
        return {"up": self.up.snapshot(), "down": self.down.snapshot()}

    def apply_event(self, message: Mapping[str, Any]) -> bool:
        event_type = str(_value(message, "event_type", "type") or "")
        payload = message.get("payload")
        data = payload if isinstance(payload, Mapping) else message
        if event_type == "book":
            outcome = self._outcome(data)
            if outcome is None:
                return False
            outcome.apply_book(data)
            tick = _decimal(_value(data, "tick_size", "tickSize"))
            if tick is not None:
                self.tick_size = tick
            return True
        if event_type == "price_change":
            changes = _value(data, "price_changes", "priceChanges")
            if not isinstance(changes, list):
                raise ValueError("malformed price_change list")
            changed = False
            for change in changes:
                if not isinstance(change, Mapping):
                    continue
                outcome = self._outcome(change)
                if outcome is None:
                    continue
                outcome.apply_price_change(change)
                changed = True
            return changed
        if event_type == "best_bid_ask":
            outcome = self._outcome(data)
            if outcome is None:
                return False
            outcome.apply_best_bid_ask(data)
            return True
        if event_type == "last_trade_price":
            outcome = self._outcome(data)
            if outcome is None:
                return False
            outcome.apply_trade(data)
            return True
        if event_type == "tick_size_change":
            tick = _decimal(_value(data, "new_tick_size", "newTickSize"))
            if tick is None:
                raise ValueError("malformed tick_size_change")
            self.tick_size = tick
            return True
        if event_type == "market_resolved":
            winning = str(
                _value(
                    data,
                    "winning_asset_id",
                    "winning_token_id",
                    "winningTokenId",
                )
                or ""
            )
            if winning == self.up_token:
                self.resolved_winner = "UP"
            elif winning == self.down_token:
                self.resolved_winner = "DOWN"
            return self.resolved_winner is not None
        return False

    def _outcome(self, data: Mapping[str, Any]) -> OutcomeState | None:
        token = str(_value(data, "asset_id", "token_id", "tokenId") or "")
        return self.by_token.get(token)


def _levels_to_map(values: Any) -> dict[Decimal, Decimal]:
    if not isinstance(values, list):
        raise ValueError("order book levels must be a list")
    result: dict[Decimal, Decimal] = {}
    for item in values:
        if not isinstance(item, Mapping):
            raise ValueError("malformed order book level")
        price = _decimal(item.get("price"))
        size = _decimal(item.get("size"))
        if price is None or size is None:
            raise ValueError("malformed order book level")
        if size > ZERO:
            result[price] = size
    return result


def _level_dict(level: Level) -> dict[str, str]:
    return {"price": str(level.price), "size": str(level.size)}


def iter_messages(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        yield payload
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                yield item
