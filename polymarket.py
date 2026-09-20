"""Minimal public Polymarket API access for 15-minute crypto markets."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

import requests


ASSET_SLUGS = {"BTC": "btc", "ETH": "eth"}
ASSET_NAMES = {"BTC": ("btc", "bitcoin"), "ETH": ("eth", "ethereum")}


def window_start_for_timestamp(timestamp: int | float) -> int:
    return math.floor(timestamp / 900) * 900


def market_slug(asset: str, window_start: int) -> str:
    asset = asset.upper()
    if asset not in ASSET_SLUGS:
        raise ValueError(f"unsupported asset: {asset}")
    if window_start % 900 != 0:
        raise ValueError("window-start must be aligned to a 15-minute UTC boundary")
    return f"{ASSET_SLUGS[asset]}-updown-15m-{window_start}"


def _parse_json_list(value: Any, field: str) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gamma malformed response: {field} is not valid JSON") from exc
    if not isinstance(value, list):
        raise RuntimeError(f"Gamma malformed response: {field} must be a list")
    return value


def map_outcome_tokens(outcomes: Any, token_ids: Any) -> dict[str, str]:
    parsed_outcomes = _parse_json_list(outcomes, "outcomes")
    parsed_tokens = _parse_json_list(token_ids, "clobTokenIds")
    if len(parsed_outcomes) != len(parsed_tokens):
        raise RuntimeError("Gamma malformed response: outcomes/token ids length mismatch")
    if len(parsed_outcomes) != 2:
        raise RuntimeError("market must contain exactly the Up and Down outcomes")

    mapping: dict[str, str] = {}
    for outcome, token_id in zip(parsed_outcomes, parsed_tokens):
        if not isinstance(outcome, str):
            raise RuntimeError("Gamma malformed response: outcome must be text")
        normalized = outcome.strip().casefold()
        if normalized in mapping:
            raise RuntimeError(f"duplicate market outcome: {outcome}")
        if not isinstance(token_id, (str, int)) or not str(token_id).strip():
            raise RuntimeError(f"missing token id for outcome: {outcome}")
        mapping[normalized] = str(token_id)

    if set(mapping) != {"up", "down"}:
        raise RuntimeError("market is missing an unambiguous Up/Down outcome mapping")
    return {"Up": mapping["up"], "Down": mapping["down"]}


def best_prices(book: dict[str, Any]) -> tuple[float | None, float | None]:
    bids = book.get("bids")
    asks = book.get("asks")
    if not isinstance(bids, list) or not isinstance(asks, list):
        raise RuntimeError("CLOB malformed response: bids and asks must be lists")
    if not bids and not asks:
        raise RuntimeError("CLOB returned a completely empty order book")
    try:
        best_bid = max(float(level["price"]) for level in bids) if bids else None
        best_ask = min(float(level["price"]) for level in asks) if asks else None
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("CLOB malformed response: invalid price level") from exc
    return best_bid, best_ask


def _parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"Gamma malformed response: missing {field}")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError as exc:
        raise RuntimeError(f"Gamma malformed response: invalid {field}") from exc


def _validate_event(
    event: dict[str, Any], asset: str, expected_slug: str, expected_start: int
) -> dict[str, Any]:
    if event.get("slug") != expected_slug:
        raise RuntimeError("Gamma returned an event with an unexpected slug")
    markets = event.get("markets")
    if not isinstance(markets, list) or not markets:
        raise RuntimeError("Gamma malformed response: event has no markets")
    exact = [m for m in markets if isinstance(m, dict) and m.get("slug") == expected_slug]
    if len(exact) != 1:
        raise RuntimeError("Gamma malformed response: expected exactly one matching market")
    market = exact[0]

    identity = " ".join(
        str(value).casefold()
        for value in (event.get("title"), market.get("question"), market.get("slug"))
        if value
    )
    if not any(name in identity for name in ASSET_NAMES[asset]):
        raise RuntimeError(f"Gamma market does not match expected asset {asset}")

    start_value = market.get("eventStartTime") or event.get("startTime")
    end_value = market.get("endDate") or event.get("endDate")
    actual_start = int(_parse_utc(start_value, "eventStartTime").timestamp())
    actual_end = int(_parse_utc(end_value, "endDate").timestamp())
    if actual_start != expected_start or actual_end != expected_start + 900:
        raise RuntimeError("Gamma market timestamps do not match the requested 15-minute window")

    tokens = map_outcome_tokens(market.get("outcomes"), market.get("clobTokenIds"))
    rules = market.get("description") or event.get("description")
    resolution_source = market.get("resolutionSource") or event.get("resolutionSource")
    if not isinstance(rules, str) or not rules.strip():
        raise RuntimeError("Gamma malformed response: missing market rules")
    resolution_text = f"{rules} {resolution_source or ''}".casefold()
    if "chainlink" not in resolution_text or "twap" not in resolution_text:
        raise RuntimeError("market resolution source is not Chainlink TWAP")
    if not isinstance(resolution_source, str) or not resolution_source.strip():
        raise RuntimeError("Gamma malformed response: missing resolution source")

    return {
        "event": event,
        "market": market,
        "tokens": tokens,
        "rules": rules,
        "resolution_source": resolution_source,
        "window_start": expected_start,
        "window_end": expected_start + 900,
    }


def fetch_market(
    session: requests.Session,
    gamma_base_url: str,
    asset: str,
    requested_start: int,
    timeout: float,
    allow_fallback: bool,
    now_timestamp: float,
) -> dict[str, Any]:
    candidates = [requested_start]
    if allow_fallback:
        candidates.extend([requested_start - 900, requested_start + 900])
    errors: list[str] = []
    for candidate in candidates:
        slug = market_slug(asset, candidate)
        try:
            response = session.get(
                f"{gamma_base_url.rstrip('/')}/events/slug/{slug}", timeout=timeout
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Gamma request error: {exc}") from exc
        if response.status_code == 404:
            errors.append(slug)
            continue
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Gamma request error: HTTP {response.status_code}") from exc
        try:
            event = response.json()
        except ValueError as exc:
            raise RuntimeError("Gamma malformed response: invalid JSON") from exc
        if not isinstance(event, dict):
            raise RuntimeError("Gamma malformed response: expected an event object")
        validated = _validate_event(event, asset, slug, candidate)
        if candidate != requested_start and not candidate <= now_timestamp <= candidate + 900:
            errors.append(slug)
            continue
        validated["slug"] = slug
        return validated
    raise RuntimeError(f"market slug not found: {market_slug(asset, requested_start)}")


def fetch_order_book(
    session: requests.Session, clob_base_url: str, token_id: str, timeout: float
) -> dict[str, Any]:
    try:
        response = session.get(
            f"{clob_base_url.rstrip('/')}/book",
            params={"token_id": token_id},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        suffix = f" HTTP {status}" if status else f" {exc}"
        raise RuntimeError(f"CLOB request error:{suffix}") from exc
    try:
        book = response.json()
    except ValueError as exc:
        raise RuntimeError("CLOB malformed response: invalid JSON") from exc
    if not isinstance(book, dict):
        raise RuntimeError("CLOB malformed response: expected an object")
    best_prices(book)
    return book
