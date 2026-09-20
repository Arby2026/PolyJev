"""Minimal OpenRouter Jev integration for snapshot probability enrichment."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from typing import Any

from openrouter import OpenRouter


PROXY_NOTICE = (
    "Binance Spot data is a predictive proxy only. "
    "The contract resolves using the Chainlink source described in the rules."
)
BLIND_INSTRUCTIONS = (
    "Based only on the provided state, will this Polymarket contract resolve UP "
    "according to its stated resolution rules?"
)
META_INSTRUCTIONS = (
    "Based on all provided information, will this Polymarket contract resolve UP "
    "according to its stated resolution rules?"
)


def concise_error(exc: Exception, limit: int = 300) -> str:
    """Return one safe, bounded line for terminal and runtime error reporting."""
    message = " ".join(str(exc).split())
    if not message:
        message = exc.__class__.__name__
    if len(message) > limit:
        return f"{message[: limit - 3]}..."
    return message


def build_blind_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset": snapshot["asset"],
        "resolution_rules": snapshot["rules"],
        "resolution_source": snapshot["resolution_source"],
        "time_remaining_sec": snapshot["time_remaining_sec"],
        "proxy_start_price": snapshot["proxy_start_price"],
        "proxy_current_price": snapshot["proxy_current_price"],
        "distance_from_start_bps": snapshot["distance_from_start_bps"],
        "return_1m": snapshot["return_1m"],
        "return_5m": snapshot["return_5m"],
        "realized_vol_5m": snapshot["realized_vol_5m"],
        "realized_vol_15m": snapshot["realized_vol_15m"],
        "proxy_notice": PROXY_NOTICE,
    }


def build_meta_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    state = build_blind_state(snapshot)
    state.update(
        {
            "p_simple": snapshot["p_simple"],
            "p_simple_description": (
                "Zero-drift probability baseline based on the Binance proxy."
            ),
            "p_market": snapshot["p_market"],
            "p_market_description": "Polymarket UP midpoint.",
            "up_best_bid": snapshot["up_best_bid"],
            "up_best_ask": snapshot["up_best_ask"],
            "down_best_bid": snapshot["down_best_bid"],
            "down_best_ask": snapshot["down_best_ask"],
            "quote_description": (
                "Current executable market quotes, not final settlement probabilities."
            ),
        }
    )
    return state


def validate_probability(value: Any) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Jev returned a non-numeric probability") from exc
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise RuntimeError("Jev probability must be between 0 and 1")
    return probability


def create_jev_client(timeout_sec: float = 9.0) -> OpenRouter:
    return OpenRouter(
        api_key=os.environ.get("OPENROUTER_API_KEY", "").strip(),
        timeout_ms=int(timeout_sec * 1000),
        retry_config=None,
    )


def call_jev(
    client: OpenRouter,
    model: str,
    state: dict[str, Any],
    instructions: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.alpha.decisions.create(
            model=model,
            state=state,
            questions={
                "resolves_up": {
                    "type": "noul",
                    "instructions": instructions,
                }
            },
        )
    except Exception as exc:
        raise RuntimeError(f"Jev request failed: {concise_error(exc)}") from exc
    latency_ms = (time.perf_counter() - started) * 1000
    try:
        answers = response.answers
        answer = answers["resolves_up"]
        value = answer.get("noul") if isinstance(answer, Mapping) else answer.noul
        probability = validate_probability(value)
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError("Jev response is missing resolves_up Noul") from exc
    usage = getattr(response, "usage", None)
    return {
        "probability": probability,
        "model": getattr(response, "model", None),
        "input_tokens": getattr(usage, "input_tokens", None),
        "latency_ms": latency_ms,
    }


def parse_choice_response(
    response: Any,
    question_name: str,
    choices: tuple[str, ...],
) -> dict[str, Any]:
    try:
        answers = response.answers
        answer = answers[question_name]
        choice = answer.get("choice") if isinstance(answer, Mapping) else answer.choice
        probabilities = (
            answer.get("probabilities")
            if isinstance(answer, Mapping)
            else answer.probabilities
        )
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Jev response is missing {question_name} Choice") from exc
    if choice not in choices or not isinstance(probabilities, Mapping):
        raise RuntimeError(f"Jev response has malformed {question_name} Choice")
    parsed: dict[str, float] = {}
    for name in choices:
        if name not in probabilities:
            raise RuntimeError(f"Jev response has malformed {question_name} probabilities")
        parsed[name] = validate_probability(probabilities[name])
    usage = getattr(response, "usage", None)
    return {
        "choice": choice,
        "probabilities": parsed,
        "model": getattr(response, "model", None),
        "input_tokens": getattr(usage, "input_tokens", None),
    }


async def call_jev_choice_async(
    client: OpenRouter,
    model: str,
    state: dict[str, Any],
    instructions: str,
    criteria: dict[str, str],
    question_name: str = "target_position",
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = await client.alpha.decisions.create_async(
            model=model,
            state=state,
            questions={
                question_name: {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": criteria,
                }
            },
        )
    except Exception as exc:
        raise RuntimeError(f"Jev request failed: {concise_error(exc)}") from exc
    result = parse_choice_response(response, question_name, tuple(criteria))
    result["latency_ms"] = (time.perf_counter() - started) * 1000
    return result
