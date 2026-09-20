"""Report position-independent forecast and terminal-edge quality from JSONL."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from live_trader import JsonlWriter, _create_http_session, _load_config
from polymarket import fetch_market, parse_resolved_outcome


HORIZONS = (15, 30, 60, 120)


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at line {line_number}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def resolve_missing_winner(
    path: Path, records: list[dict[str, Any]]
) -> str | None:
    resolved = [item for item in records if item.get("record_type") == "market_resolved"]
    if resolved:
        return resolved[-1].get("winner")
    start = next(
        (item for item in records if item.get("record_type") == "session_start"), None
    )
    if start is None:
        return None
    config = _load_config()
    session = _create_http_session(int(config["http_retries"]))
    fresh = fetch_market(
        session,
        str(config["gamma_base_url"]),
        str(start["asset"]),
        int(start["window_start"]),
        float(config["http_timeout_sec"]),
        False,
        time.time(),
    )
    market = fresh["market"]
    if market.get("closed") is not True and fresh["event"].get("closed") is not True:
        return None
    winner = parse_resolved_outcome(market.get("outcomes"), market.get("outcomePrices"))
    if winner is not None:
        JsonlWriter(path).write(
            "market_resolved",
            resolved_at_report_time=True,
            winner=winner,
        )
        records.append({"record_type": "market_resolved", "winner": winner})
    return winner


def generate_report(path: Path, resolve_missing: bool = True) -> str:
    records = load_records(path)
    forecasts = {
        item["decision_id"]: item
        for item in records
        if item.get("record_type") == "forecast" and item.get("decision_id")
    }
    labels = [item for item in records if item.get("record_type") == "label"]
    winner = None
    for item in records:
        if item.get("record_type") == "market_resolved":
            winner = item.get("winner")
    if winner is None and resolve_missing:
        winner = resolve_missing_winner(path, records)
    lines = ["Forecast report", "===============", "", "Short-horizon labels"]
    for horizon in HORIZONS:
        for side in ("UP", "DOWN"):
            group = [
                label
                for label in labels
                if label.get("horizon") == horizon and label.get("side") == side
            ]
            valid = [label for label in group if label.get("label_status") == "valid"]
            unexecutable = sum(
                label.get("label_status") == "unexecutable" for label in group
            )
            stale = sum(label.get("label_status") == "stale" for label in group)
            probabilities = [_float(item.get("predicted_probability")) for item in valid]
            outcomes = [1.0 if item.get("profitable") else 0.0 for item in valid]
            pnls = [_float(item.get("actual_net_pnl")) for item in valid]
            spreads = [
                _forecast_spread(forecasts.get(item.get("decision_id")), side)
                for item in valid
            ]
            ages = [_float(item.get("book_age_ms")) for item in valid]
            lines.append(
                f"{horizon:>3}s {side}: valid={len(valid)} unexecutable={unexecutable} "
                f"stale={stale} mean_p={_fmt(_mean(probabilities))} "
                f"actual={_fmt(_mean(outcomes))} brier={_fmt(_brier(probabilities, outcomes))} "
                f"mean_pnl={_fmt(_mean(pnls), 5)} median_pnl={_fmt(_median(pnls), 5)} "
                f"avg_spread_ticks={_fmt(_mean(_present(spreads)))} "
                f"avg_book_age_ms={_fmt(_mean(_present(ages)), 1)}"
            )
    lines.extend(["", "Probability diagnostics (valid labels, overlapping forecasts)"])
    bins = [
        ("p < .40", lambda value: value < 0.40),
        (".40 <= p < .60", lambda value: 0.40 <= value < 0.60),
        (".60 <= p < .75", lambda value: 0.60 <= value < 0.75),
        ("p >= .75", lambda value: value >= 0.75),
    ]
    valid_all = [item for item in labels if item.get("label_status") == "valid"]
    for name, predicate in bins:
        group = [item for item in valid_all if predicate(_float(item["predicted_probability"]))]
        actual = [1.0 if item.get("profitable") else 0.0 for item in group]
        pnls = [_float(item.get("actual_net_pnl")) for item in group]
        lines.append(
            f"{name}: N={len(group)} actual_profitable={_fmt(_mean(actual))} "
            f"mean_net_pnl={_fmt(_mean(pnls), 5)}"
        )
    lines.extend(["", "Terminal resolution trajectory"])
    if winner not in ("UP", "DOWN"):
        lines.append("winner: settlement pending")
    else:
        lines.append(f"winner: {winner}")
        lines.append("Terminal observations within one market are highly correlated.")
        terminal = list(forecasts.values())
        for minutes in (12, 9, 6, 3, 1):
            if terminal:
                selected = min(
                    terminal,
                    key=lambda item: abs(int(item.get("time_remaining", 0)) - minutes * 60),
                )
                lines.append(
                    f"nearest {minutes}m: {_terminal_line(selected, winner, winner)}"
                )
        winner_edge = f"model_terminal_edge_{winner.lower()}"
        loser = "DOWN" if winner == "UP" else "UP"
        loser_edge = f"model_terminal_edge_{loser.lower()}"
        positive_winner = sorted(
            [item for item in terminal if _nullable_float(item.get(winner_edge), -1) > 0],
            key=lambda item: _nullable_float(item.get(winner_edge), -1),
            reverse=True,
        )[:10]
        lines.append("")
        lines.append("TOP positive terminal edges on WINNER side")
        lines.extend(_terminal_line(item, winner, winner) for item in positive_winner)
        positive_loser = sorted(
            [item for item in terminal if _nullable_float(item.get(loser_edge), -1) > 0],
            key=lambda item: _nullable_float(item.get(loser_edge), -1),
            reverse=True,
        )[:10]
        lines.append("")
        lines.append("TOP positive terminal edges on LOSING side")
        lines.extend(_terminal_line(item, loser, winner) for item in positive_loser)
        earliest = max(
            (
                item
                for item in terminal
                if _nullable_float(item.get(winner_edge), -1) > 0
            ),
            key=lambda item: int(item.get("time_remaining", 0)),
            default=None,
        )
        lines.append("")
        lines.append(
            "earliest winner-side edge > 0: "
            + (_terminal_line(earliest, winner, winner) if earliest else "none")
        )
    return "\n".join(lines)


def _terminal_line(
    record: dict[str, Any] | None, side: str, actual_winner: str
) -> str:
    if record is None:
        return "none"
    key = side.lower()
    entry = record.get("hypothetical_entries", {}).get(side, {})
    p = record.get("p_resolve_up")
    probability = _float(p) if side == "UP" else 1 - _float(p)
    break_even = record.get(f"{key}_terminal_break_even")
    edge = record.get(f"model_terminal_edge_{key}")
    ask = record.get("compact_state_sent_to_jev", {}).get(key, {}).get("ask")
    hold_pnl = None
    try:
        shares = float(entry["shares"])
        cost = float(entry["net_cash"])
        hold_pnl = shares - cost if side == actual_winner else -cost
    except (KeyError, TypeError, ValueError):
        pass
    return (
        f"left={record.get('time_remaining')}s side={side} ask={ask} "
        f"break_even={break_even} p={probability:.4f} edge={edge} "
        f"hold_pnl={_fmt(hold_pnl, 5)}"
    )
def _forecast_spread(forecast: dict[str, Any] | None, side: str) -> float | None:
    if not forecast:
        return None
    try:
        return float(forecast["clob_liquidity_summary"][side.lower()]["spread_ticks"])
    except (KeyError, TypeError, ValueError):
        return None


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def _median(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.median(items) if items else None


def _brier(probabilities: list[float], outcomes: list[float]) -> float | None:
    if not probabilities:
        return None
    return statistics.fmean((p - y) ** 2 for p, y in zip(probabilities, outcomes))


def _present(values: Iterable[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def _float(value: Any) -> float:
    return float(value)


def _nullable_float(value: Any, default: float) -> float:
    return default if value is None else float(value)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Report a Jev forecast JSONL session")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        print(generate_report(args.path))
        return 0
    except Exception as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
