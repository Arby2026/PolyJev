"""Report position-independent forecast and terminal-edge quality from JSONL."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from live_trader import JsonlWriter, _create_http_session, _load_config
from polymarket import fetch_market, parse_resolved_outcome


HORIZONS = (15, 30, 60, 120)


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if path.suffix.casefold() == ".zip":
        archive = zipfile.ZipFile(path)
        names = [name for name in archive.namelist() if name.endswith(".jsonl")]
        if len(names) != 1:
            raise RuntimeError("forecast ZIP must contain exactly one JSONL file")
        handle = archive.open(names[0], "r")
        lines = (raw.decode("utf-8") for raw in handle)
    else:
        archive = None
        handle = path.open("r", encoding="utf-8")
        lines = handle
    try:
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at line {line_number}") from exc
            if isinstance(record, dict):
                records.append(record)
    finally:
        handle.close()
        if archive is not None:
            archive.close()
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
    if winner is not None and path.suffix.casefold() != ".zip":
        JsonlWriter(path).write(
            "market_resolved",
            resolved_at_report_time=True,
            winner=winner,
        )
        records.append({"record_type": "market_resolved", "winner": winner})
    return winner


def generate_report(path: Path, resolve_missing: bool = True) -> str:
    records = load_records(path)
    is_v2 = any(
        int(item.get("experiment_version", 1)) >= 2
        for item in records
    )
    if is_v2:
        return _generate_v2_report(path, records, resolve_missing)
    report = _generate_legacy_report(path, records, resolve_missing)
    return report + "\n\nactionable/latency-adjusted metrics = unavailable"


def _generate_v2_report(
    path: Path, records: list[dict[str, Any]], resolve_missing: bool
) -> str:
    forecasts = [item for item in records if item.get("record_type") == "forecast" and not item.get("skipped")]
    by_id = {item["decision_id"]: item for item in forecasts}
    labels = [item for item in records if item.get("record_type") == "label"]
    crossovers = [item for item in records if item.get("record_type") == "crossover"]
    winner = next(
        (
            item.get("winner")
            for item in reversed(records)
            if item.get("record_type") == "market_resolved"
        ),
        None,
    )
    if winner is None and resolve_missing:
        winner = resolve_missing_winner(path, records)
    version = max((int(item.get("experiment_version", 1)) for item in records), default=2)
    lines = [
        f"Forecast Experiment V{version} — side-neutral report",
        "============================================",
        f"winner_side: {winner or 'settlement pending'}",
        "Observations within one market are highly correlated and are not independent market trials.",
    ]
    lines.extend(_feature_diagnostics(records, forecasts, winner))
    if winner not in ("UP", "DOWN"):
        return "\n".join(lines)
    loser = "DOWN" if winner == "UP" else "UP"
    lines.extend(["", "Winner-relative time buckets"])
    buckets = [
        (720, 900, "12-15m"),
        (540, 720, "9-12m"),
        (360, 540, "6-9m"),
        (180, 360, "3-6m"),
        (0, 180, "0-3m"),
    ]
    for low, high, name in buckets:
        group = [
            item
            for item in forecasts
            if low <= int(item.get("time_remaining", -1)) < high
            or (high == 900 and int(item.get("time_remaining", -1)) == 900)
        ]
        metrics = [_winner_metrics(item, winner) for item in group]
        jev_winner = [item["p_jev_winner"] for item in metrics]
        market_winner = [item["p_market_winner"] for item in metrics]
        jev_brier = [item["jev_brier"] for item in metrics]
        market_brier = [item["market_brier"] for item in metrics]
        abs_delta = [
            abs(float(item["p_jev_up"]) - _response_market_up(item))
            for item in group
        ]
        best = Counter(item.get("best_edge_side", "NONE") for item in group)
        lines.append(
            f"{name}: N={len(group)} p_jev_winner mean/min/max="
            f"{_triple(jev_winner)} p_market_winner={_triple(market_winner)} "
            f"Jev_Brier={_fmt(_mean(jev_brier))} Market_Brier={_fmt(_mean(market_brier))} "
            f"Brier_delta={_fmt(_difference(_mean(market_brier), _mean(jev_brier)))} "
            f"mean_abs_Jev-Market={_fmt(_mean(abs_delta))}"
        )
        lines.append(
            f"  directional: Jev={_match_rate(group, 'jev_side', winner)} "
            f"Market={_match_rate(group, 'market_side', winner)} "
            f"RAW={_match_rate(group, 'raw_side', winner)} "
            f"TWAP={_match_rate(group, 'twap_side', winner)}; "
            f"best_edge winner={_ratio(best[winner], len(group))} "
            f"loser={_ratio(best[loser], len(group))} NONE={_ratio(best['NONE'], len(group))}"
        )

    lines.extend(["", "ACTIONABLE / POST-INFERENCE terminal edges"])
    classifications = Counter(
        _edge_classification(item, winner) for item in forecasts
    )
    for name in ("WINNER_EDGE_ONLY", "LOSER_EDGE_ONLY", "BOTH", "NEITHER"):
        lines.append(f"{name}: {classifications[name]}")
    lines.extend(_edge_extremes(forecasts, winner, "winner_side", winner))
    lines.extend(_edge_extremes(forecasts, loser, "loser_side", winner))
    lines.extend(["", "ORACLE / PRE-INFERENCE edges are logged but are not tradable."])

    lines.extend(["", "Crossovers and lead/lag"])
    for source in ("RAW", "TWAP", "JEV", "MARKET"):
        source_events = [item for item in crossovers if item.get("source") == source]
        lines.append(f"{source}: transitions={len(source_events)}")
    for direction in (("UP", "DOWN"), ("DOWN", "UP")):
        for other in ("RAW", "TWAP", "MARKET"):
            lags = _nearest_transition_lags(crossovers, "JEV", other, direction)
            lines.append(
                f"JEV vs {other} {direction[0]}->{direction[1]}: "
                f"N={len(lags)} mean_other_minus_jev_s={_fmt(_mean(lags), 2)}"
            )

    lines.extend(["", "Short-horizon repricing and execution diagnostics"])
    disagreement_bins = [
        (0.0, 0.05, "<5pp"),
        (0.05, 0.10, "5-10pp"),
        (0.10, 0.20, "10-20pp"),
        (0.20, float("inf"), ">=20pp"),
    ]
    for horizon in HORIZONS:
        horizon_labels = [
            item
            for item in labels
            if item.get("horizon") == horizon and item.get("label_status") == "valid"
        ]
        lines.append(f"{horizon}s:")
        for low, high, name in disagreement_bins:
            group = []
            for label in horizon_labels:
                forecast = by_id.get(label.get("decision_id"))
                if forecast is None:
                    continue
                magnitude = abs(float(forecast["p_jev_up"]) - _response_market_up(forecast))
                if low <= magnitude < high:
                    group.append((label, forecast))
            repricings = [_nullable_float(item[0].get("signed_repricing"), 0) for item in group]
            pnls = [_disagreement_pnl(label, forecast) for label, forecast in group]
            lines.append(
                f"  {name}: N={len(group)} mean_signed_repricing={_fmt(_mean(repricings))} "
                f"median={_fmt(_median(repricings))} "
                f"mean_disagreement_side_pnl={_fmt(_mean(_present(pnls)), 5)}"
            )
        jev_repricing = [
            _nullable_float(item.get("signed_repricing"), 0) for item in horizon_labels
        ]
        twap_repricing = [
            _twap_signed_repricing(item, by_id.get(item.get("decision_id")))
            for item in horizon_labels
        ]
        lines.append(
            f"  baseline mean signed repricing: Jev={_fmt(_mean(jev_repricing))} "
            f"TWAP-side={_fmt(_mean(_present(twap_repricing)))}"
        )

    lines.extend(["", "Jev conditioned on Chainlink TWAP side"])
    for same, label_name in ((True, "Jev side == TWAP side"), (False, "Jev side != TWAP side")):
        group = [
            item
            for item in forecasts
            if (item.get("jev_side") == item.get("twap_side")) is same
        ]
        group_ids = {item["decision_id"] for item in group}
        group_labels = [
            item
            for item in labels
            if item.get("decision_id") in group_ids and item.get("label_status") == "valid"
        ]
        lines.append(
            f"{label_name}: N={len(group)} future_signed_repricing="
            f"{_fmt(_mean([_nullable_float(item.get('signed_repricing'), 0) for item in group_labels]))} "
            f"winner_match={_match_rate(group, 'jev_side', winner)} "
            f"mean_abs_delta={_fmt(_mean([abs(float(item['p_jev_up']) - _response_market_up(item)) for item in group]))}"
        )
    return "\n".join(lines)


def pearson(pairs: Iterable[tuple[Any, Any]]) -> tuple[float | None, int]:
    values = []
    for left, right in pairs:
        try:
            x, y = float(left), float(right)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            values.append((x, y))
    if len(values) < 3:
        return None, len(values)
    xs, ys = zip(*values)
    if min(xs) == max(xs) or min(ys) == max(ys):
        return None, len(values)
    return statistics.correlation(xs, ys), len(values)


def _input_feature(record: dict[str, Any], key: str) -> Any:
    if key in record:
        return record[key]
    # V2 uses the pre-inference snapshot too; never correlate with later prices.
    state = record.get("input_state", {})
    if key == "distance_from_start_bps":
        return state.get("chainlink", {}).get("raw_from_open_bps")
    if key == "time_left_sec":
        return state.get("time_remaining")
    return None


def _feature_diagnostics(records: list[dict[str, Any]], forecasts: list[dict[str, Any]], winner: str | None) -> list[str]:
    candidates = [r for r in records if r.get("record_type") in ("forecast", "forecast_skip")]
    skipped = [r for r in candidates if r.get("skipped")]
    percentage = 100 * len(skipped) / len(candidates) if candidates else None
    lines = [
        "", f"skipped: {len(skipped)}/{len(candidates)} ({_fmt(percentage, 2)}%)",
        "Skip denominator: logged candidates; at most one per second, coalesced during inference.",
    ]
    for reason, count in sorted(Counter(str(r.get("skip_reason")) for r in skipped).items()):
        lines.append(f"  {reason}: {count}")
    lines.append("Pearson correlations (Q1 = p_final_up; V2 alias p_jev_up; input-time features):")
    for key in ("distance_from_start_bps", "time_left_sec"):
        correlation, count = pearson(
            (r.get("p_final_up", r.get("p_jev_up")), _input_feature(r, key)) for r in forecasts
        )
        target = "unavailable" if correlation is None else ("PASS" if abs(correlation) < 0.6 else "FAIL")
        lines.append(f"correlation Q1 vs {key}: r={_fmt(correlation)} N={count} target |r|<0.6: {target}")
    lines.append("Correlations are descriptive; one market does not establish a general improvement.")
    v3 = [r for r in candidates if int(r.get("experiment_version", 1)) >= 3]
    if v3:
        lines.append("V3 input features (including skips), mean/min/max:")
        for key in ("time_left_sec", "distance_from_start_bps", "mid", "spread_bps", "depth_ratio", "vol_60s", "fees_bps"):
            values = [float(r[key]) for r in v3 if r.get(key) is not None]
            lines.append(f"  {key}: N={len(values)} {_triple(values)}")
        walls = sum(r.get("wall_text") not in (None, "none detected") for r in v3)
        lines.append(f"  wall_text: detected in {walls}/{len(v3)} candidates")
        lines.append(f"  vol_regime: {dict(Counter(r.get('vol_regime', 'unavailable') for r in v3))}")
    lines.append("Flow 60s when Jev correct/incorrect (BUY UP USD minus BUY DOWN USD):")
    if winner not in ("UP", "DOWN"):
        lines.append("  settlement pending; correctness unavailable")
        return lines
    for correct, label in ((True, "correct"), (False, "incorrect")):
        group = [r for r in forecasts if (
            ("UP" if float(r.get("p_final_up", r.get("p_jev_up"))) >= 0.5 else "DOWN") == winner
        ) == correct]
        flows = [r["flow_60s"] for r in group if r.get("flow_60s", {}).get("status") == "ready"]
        values = [float(f["imbalance_usd"]) for f in flows]
        bins = [sum(v < -100 for v in values), sum(-100 <= v < 0 for v in values), sum(v == 0 for v in values), sum(0 < v <= 100 for v in values), sum(v > 100 for v in values)]
        lines.append(
            f"  {label}: N={len(flows)} unavailable_or_partial={len(group) - len(flows)} "
            f"imbalance_usd mean/min/max={_triple(values)} median={_fmt(_median(values))} "
            f"bins [<-100, -100..0, zero, 0..100, >100]={bins}"
        )
        for key in ("up_count", "down_count", "avg_size", "net"):
            lines.append(f"    {key}: {_triple([float(f[key]) for f in flows])}")
    return lines


def _winner_metrics(record: dict[str, Any], winner: str) -> dict[str, float]:
    p_jev_up = float(record["p_jev_up"])
    p_market_up = _response_market_up(record)
    y = 1.0 if winner == "UP" else 0.0
    return {
        "p_jev_winner": p_jev_up if winner == "UP" else 1 - p_jev_up,
        "p_market_winner": p_market_up if winner == "UP" else 1 - p_market_up,
        "jev_brier": (p_jev_up - y) ** 2,
        "market_brier": (p_market_up - y) ** 2,
    }


def _response_market_up(record: dict[str, Any]) -> float:
    return float(record["response_market_probabilities"]["p_market_up"])


def _edge_classification(record: dict[str, Any], winner: str) -> str:
    up = _nullable_float(record.get("actionable_edge_up"), float("-inf"))
    down = _nullable_float(record.get("actionable_edge_down"), float("-inf"))
    winner_positive = (up if winner == "UP" else down) > 0
    loser_positive = (down if winner == "UP" else up) > 0
    if winner_positive and loser_positive:
        return "BOTH"
    if winner_positive:
        return "WINNER_EDGE_ONLY"
    if loser_positive:
        return "LOSER_EDGE_ONLY"
    return "NEITHER"


def _edge_extremes(
    forecasts: list[dict[str, Any]], side: str, label: str, winner: str
) -> list[str]:
    key = f"actionable_edge_{side.lower()}"
    positive = [item for item in forecasts if _nullable_float(item.get(key), -1) > 0]
    earliest = max(positive, key=lambda item: int(item.get("time_remaining", 0)), default=None)
    largest = max(positive, key=lambda item: _nullable_float(item.get(key), -1), default=None)
    run = _longest_positive_run(forecasts, key)
    return [
        f"earliest {label} edge > 0: {_v2_edge_line(earliest, side, winner)}",
        f"largest {label} edge: {_v2_edge_line(largest, side, winner)}",
        f"longest continuous {label} edge run: {run}",
    ]


def _v2_edge_line(
    record: dict[str, Any] | None, side: str, winner: str
) -> str:
    if record is None:
        return "none"
    key = side.lower()
    state = record.get("response_state", {})
    entry = record.get("response_entries", {}).get(side, {})
    probability = float(record["p_jev_up"])
    if side == "DOWN":
        probability = 1 - probability
    chain = state.get("chainlink", {})
    pnl = None
    try:
        pnl = (
            float(entry["shares"]) - float(entry["net_cash"])
            if side == winner
            else -float(entry["net_cash"])
        )
    except (KeyError, TypeError, ValueError):
        pass
    return (
        f"left={record.get('time_remaining')}s side={side} p_jev_side={probability:.4f} "
        f"bid/ask={state.get(key, {}).get('bid')}/{state.get(key, {}).get('ask')} "
        f"break_even={record.get('terminal_break_even_' + key)} "
        f"edge={record.get('actionable_edge_' + key)} "
        f"raw_bps={chain.get('raw_from_open_bps')} twap_bps={chain.get('twap_from_open_bps')} "
        f"winner_hold_pnl={_fmt(pnl, 5)}"
    )


def _longest_positive_run(forecasts: list[dict[str, Any]], key: str) -> str:
    ordered = sorted(forecasts, key=lambda item: float(item.get("response_timestamp", 0)))
    best: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for item in ordered:
        if _nullable_float(item.get(key), -1) > 0:
            current.append(item)
            if len(current) > len(best):
                best = list(current)
        else:
            current = []
    if not best:
        return "none"
    return (
        f"N={len(best)} left={best[0].get('time_remaining')}s"
        f"->{best[-1].get('time_remaining')}s"
    )


def _nearest_transition_lags(
    events: list[dict[str, Any]], source: str, other: str, direction: tuple[str, str]
) -> list[float]:
    first = [
        item
        for item in events
        if item.get("source") == source
        and (item.get("from_side"), item.get("to_side")) == direction
    ]
    second = [
        item
        for item in events
        if item.get("source") == other
        and (item.get("from_side"), item.get("to_side")) == direction
    ]
    return [
        float(min(second, key=lambda other_item: abs(float(other_item["timestamp"]) - float(item["timestamp"])), default={"timestamp": item["timestamp"]})["timestamp"])
        - float(item["timestamp"])
        for item in first
        if second
    ]


def _disagreement_pnl(label: dict[str, Any], forecast: dict[str, Any]) -> float | None:
    side = "up" if float(forecast["p_jev_up"]) > _response_market_up(forecast) else "down"
    value = label.get(f"{side}_response_entry_result", {}).get("net_pnl")
    return None if value is None else float(value)


def _twap_signed_repricing(
    label: dict[str, Any], forecast: dict[str, Any] | None
) -> float | None:
    if forecast is None or label.get("future_p_market_up") is None:
        return None
    change = float(label["future_p_market_up"]) - _response_market_up(forecast)
    return change if forecast.get("twap_side") == "UP" else -change


def _match_rate(group: list[dict[str, Any]], key: str, winner: str) -> str:
    return _ratio(sum(item.get(key) == winner for item in group), len(group))


def _ratio(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{numerator / denominator:.3f}"


def _triple(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"{statistics.fmean(values):.4f}/{min(values):.4f}/{max(values):.4f}"


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _generate_legacy_report(
    path: Path,
    records: list[dict[str, Any]],
    resolve_missing: bool = True,
) -> str:
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
