"""Binance Spot public REST data used only as a predictive proxy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}


@dataclass(frozen=True)
class Candle:
    open_time_ms: int
    close_time_ms: int
    open: float
    close: float


def fetch_binance_data(
    session: requests.Session,
    base_url: str,
    asset: str,
    window_start: int,
    observed_at_timestamp: float,
    timeout: float,
) -> tuple[float, list[Candle]]:
    symbol = SYMBOLS[asset]
    try:
        ticker_response = session.get(
            f"{base_url.rstrip('/')}/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=timeout,
        )
        ticker_response.raise_for_status()
        start_ms = int(min(window_start - 120, observed_at_timestamp - 30 * 60) * 1000)
        candles_response = session.get(
            f"{base_url.rstrip('/')}/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": "1m",
                "startTime": start_ms,
                "endTime": int(observed_at_timestamp * 1000),
                "limit": 100,
            },
            timeout=timeout,
        )
        candles_response.raise_for_status()
        ticker_payload: Any = ticker_response.json()
        candles_payload: Any = candles_response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"Binance unavailable: {exc}") from exc

    try:
        current_price = float(ticker_payload["price"])
        candles = [
            Candle(
                open_time_ms=int(row[0]),
                close_time_ms=int(row[6]),
                open=float(row[1]),
                close=float(row[4]),
            )
            for row in candles_payload
        ]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("Binance unavailable: malformed response") from exc
    if current_price <= 0 or any(c.open <= 0 or c.close <= 0 for c in candles):
        raise RuntimeError("Binance unavailable: non-positive price")
    if len(candles) < 17:
        raise RuntimeError("insufficient candles from Binance")
    return current_price, candles

