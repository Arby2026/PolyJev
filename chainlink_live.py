"""Minimal Polymarket RTDS Chainlink parsing and rolling live state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Literal, Mapping


RTDS_URL = "wss://ws-live-data.polymarket.com"
RAW_TOPIC = "crypto_prices_chainlink"
TWAP60_TOPIC = "crypto_prices_twap_sixty"
E18 = Decimal("1000000000000000000")
ZERO = Decimal("0")


@dataclass(frozen=True)
class ChainlinkUpdate:
    kind: Literal["raw", "twap60"]
    symbol: str
    timestamp_ms: int
    value: Decimal


@dataclass(frozen=True)
class OpeningReference:
    price: Decimal
    source_timestamp_ms: int
    offset_ms: int
    quality: Literal["good", "poor"]


def subscription_frame(topic: str) -> dict[str, Any]:
    return {
        "action": "subscribe",
        "subscriptions": [{"topic": topic, "type": "update"}],
    }


def parse_rtds_event(
    message: Mapping[str, Any], expected_symbol: str
) -> ChainlinkUpdate | None:
    """Parse only the requested Chainlink raw/TWAP60 wire events."""
    if message.get("type") != "update":
        return None
    topic = message.get("topic")
    if topic not in (RAW_TOPIC, TWAP60_TOPIC):
        return None
    payload = message.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("RTDS event is missing payload")
    symbol = str(payload.get("symbol") or "").casefold()
    if symbol != expected_symbol.casefold():
        return None
    timestamp_ms = _timestamp_ms(payload.get("timestamp"))
    if topic == RAW_TOPIC:
        return ChainlinkUpdate(
            "raw", symbol, timestamp_ms, _decimal(payload.get("value"), "value")
        )
    window = payload.get("window_s")
    if isinstance(window, bool) or window != 60:
        return None
    if payload.get("full_accuracy_value") is not None:
        full = str(payload["full_accuracy_value"])
        if not full or full.lstrip("-").isdigit() is False:
            raise ValueError("full_accuracy_value must be an integer string")
        value = Decimal(full) / E18
    else:
        value = _decimal(payload.get("value"), "value")
    return ChainlinkUpdate("twap60", symbol, timestamp_ms, value)


def choose_opening_reference(
    updates: Iterable[ChainlinkUpdate], window_start_ms: int
) -> OpeningReference | None:
    raw = [update for update in updates if update.kind == "raw"]
    if not raw:
        return None
    closest = min(raw, key=lambda item: abs(item.timestamp_ms - window_start_ms))
    offset = closest.timestamp_ms - window_start_ms
    return OpeningReference(
        price=closest.value,
        source_timestamp_ms=closest.timestamp_ms,
        offset_ms=offset,
        quality="good" if abs(offset) <= 5000 else "poor",
    )


class ChainlinkLiveState:
    def __init__(self, symbol: str):
        self.symbol = symbol.casefold()
        self.raw_updates: deque[ChainlinkUpdate] = deque(maxlen=100)
        self.twap_updates: deque[ChainlinkUpdate] = deque(maxlen=100)
        self.opening: OpeningReference | None = None
        self.raw_events = 0
        self.twap_events = 0

    def apply(self, message: Mapping[str, Any]) -> bool:
        update = parse_rtds_event(message, self.symbol)
        if update is None:
            return False
        if update.kind == "raw":
            self.raw_updates.append(update)
            self.raw_events += 1
        else:
            self.twap_updates.append(update)
            self.twap_events += 1
        return True

    def capture_opening(self, window_start_ms: int) -> OpeningReference | None:
        self.opening = choose_opening_reference(self.raw_updates, window_start_ms)
        return self.opening

    @property
    def ready(self) -> bool:
        return bool(self.opening and self.raw_updates and self.twap_updates)

    def compact(self) -> dict[str, Any]:
        if not self.ready or self.opening is None:
            raise RuntimeError("Chainlink state is not ready")
        raw_values = [update.value for update in self.raw_updates]
        twap_values = [update.value for update in self.twap_updates]
        raw = raw_values[-1]
        twap = twap_values[-1]
        opening = self.opening.price
        return {
            "open": _compact_decimal(opening, 8),
            "raw": _compact_decimal(raw, 8),
            "twap60": _compact_decimal(twap, 8),
            "raw_from_open_bps": _bps(raw, opening),
            "twap_from_open_bps": _bps(twap, opening),
            "raw_d1": _history_bps(raw_values, 1),
            "raw_d5": _history_bps(raw_values, 5),
            "raw_d20": _history_bps(raw_values, 20),
            "twap_d1": _history_bps(twap_values, 1),
            "twap_d5": _history_bps(twap_values, 5),
            "twap_d20": _history_bps(twap_values, 20),
            "twap_recent": [_compact_decimal(value, 8) for value in twap_values[-5:]],
        }

    def logger_snapshot(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "opening": None if self.opening is None else {
                "price": str(self.opening.price),
                "source_timestamp_ms": self.opening.source_timestamp_ms,
                "offset_ms": self.opening.offset_ms,
                "quality": self.opening.quality,
            },
            "current_raw": str(self.raw_updates[-1].value) if self.raw_updates else None,
            "current_raw_timestamp_ms": (
                self.raw_updates[-1].timestamp_ms if self.raw_updates else None
            ),
            "current_twap60": (
                str(self.twap_updates[-1].value) if self.twap_updates else None
            ),
            "current_twap_timestamp_ms": (
                self.twap_updates[-1].timestamp_ms if self.twap_updates else None
            ),
        }


def _timestamp_ms(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("invalid Chainlink timestamp")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid Chainlink timestamp") from exc
    return parsed * 1000 if abs(parsed) < 100_000_000_000 else parsed


def _decimal(value: Any, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"invalid Chainlink {field}") from exc
    if not parsed.is_finite():
        raise ValueError(f"invalid Chainlink {field}")
    return parsed


def _bps(current: Decimal, old: Decimal) -> float | None:
    if old == ZERO:
        return None
    return round(float((current - old) / old * Decimal("10000")), 2)


def _history_bps(values: list[Decimal], periods: int) -> float | None:
    if len(values) <= periods:
        return None
    return _bps(values[-1], values[-1 - periods])


def _compact_decimal(value: Decimal, places: int) -> str:
    quantum = Decimal(1).scaleb(-places)
    text = format(value.quantize(quantum), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text
