"""Pure feature calculations for the Binance proxy series."""

from __future__ import annotations

import math
import statistics

from market_data import Candle


def _closest_close(candles: list[Candle], target_timestamp: float) -> float:
    candle = min(candles, key=lambda c: abs(c.close_time_ms / 1000 - target_timestamp))
    return candle.close


def calculate_features(
    candles: list[Candle],
    current_price: float,
    window_start: int,
    window_end: int,
    observed_at_timestamp: float,
) -> dict[str, float]:
    completed = sorted(
        (c for c in candles if c.close_time_ms < observed_at_timestamp * 1000),
        key=lambda c: c.close_time_ms,
    )
    if len(completed) < 16:
        raise RuntimeError("insufficient candles: need 16 completed 1-minute candles")

    start_candle = min(candles, key=lambda c: abs(c.open_time_ms / 1000 - window_start))
    if abs(start_candle.open_time_ms / 1000 - window_start) > 60:
        raise RuntimeError("insufficient candles: no price near window start")
    proxy_start_price = start_candle.open

    def realized_vol(intervals: int) -> float:
        closes = [c.close for c in completed[-(intervals + 1) :]]
        if len(closes) != intervals + 1:
            raise RuntimeError(f"insufficient candles for {intervals}m volatility")
        log_returns = [math.log(b / a) for a, b in zip(closes, closes[1:])]
        return statistics.pstdev(log_returns)

    price_1m_ago = _closest_close(completed, observed_at_timestamp - 60)
    price_5m_ago = _closest_close(completed, observed_at_timestamp - 300)
    return {
        "proxy_start_price": proxy_start_price,
        "proxy_current_price": current_price,
        "distance_from_start_bps": (current_price / proxy_start_price - 1) * 10_000,
        "return_1m": current_price / price_1m_ago - 1,
        "return_5m": current_price / price_5m_ago - 1,
        "realized_vol_5m": realized_vol(5),
        "realized_vol_15m": realized_vol(15),
        "time_remaining_sec": window_end - observed_at_timestamp,
    }


def calculate_p_simple(
    proxy_start_price: float,
    proxy_current_price: float,
    realized_vol_15m: float,
    time_remaining_sec: float,
) -> float | None:
    """Zero-drift baseline based only on the Binance proxy and minute volatility."""
    remaining_minutes = time_remaining_sec / 60
    if remaining_minutes <= 0:
        return None
    if proxy_start_price <= 0 or proxy_current_price <= 0:
        raise RuntimeError("P_simple requires positive proxy prices")
    if abs(realized_vol_15m) <= 1e-12:
        if proxy_current_price > proxy_start_price:
            return 1.0
        if proxy_current_price < proxy_start_price:
            return 0.0
        return 0.5
    distance = math.log(proxy_current_price / proxy_start_price)
    z_score = distance / (realized_vol_15m * math.sqrt(remaining_minutes))
    return 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
