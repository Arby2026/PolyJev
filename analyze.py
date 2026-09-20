"""Stage 4 falsification analysis for resolved PolyJev checkpoints."""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import yaml


EPSILON = 1e-12
CHECKPOINTS = ("T-10", "T-5", "T-2")
MODELS = (
    ("Market", "p_market"),
    ("Simple", "p_simple"),
    ("Jev Blind", "p_jev_blind"),
    ("Jev Meta", "p_jev_meta"),
)


class AnalysisError(RuntimeError):
    """Raised when stored observations cannot be analyzed safely."""


def outcome_to_binary(outcome: str) -> int:
    if outcome == "UP":
        return 1
    if outcome == "DOWN":
        return 0
    raise ValueError(f"invalid outcome: {outcome!r}")


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    _validate_metric_inputs(probabilities, outcomes)
    return sum((probability - outcome) ** 2 for probability, outcome in zip(probabilities, outcomes)) / len(outcomes)


def binary_log_loss(
    probabilities: Sequence[float], outcomes: Sequence[int], epsilon: float = EPSILON
) -> float:
    _validate_metric_inputs(probabilities, outcomes)
    total = 0.0
    for probability, outcome in zip(probabilities, outcomes):
        clamped = min(max(probability, epsilon), 1 - epsilon)
        total -= outcome * math.log(clamped) + (1 - outcome) * math.log(1 - clamped)
    return total / len(outcomes)


def disagreement_metrics(
    market_probabilities: Sequence[float],
    jev_probabilities: Sequence[float],
    outcomes: Sequence[int],
    threshold: float,
) -> dict[str, float | int | None]:
    if not (
        len(market_probabilities) == len(jev_probabilities) == len(outcomes)
    ):
        raise ValueError("probabilities and outcomes must have equal lengths")
    selected = [
        index
        for index, (market, jev) in enumerate(
            zip(market_probabilities, jev_probabilities)
        )
        if abs(jev - market) >= threshold - EPSILON
    ]
    if not selected:
        return {
            "n": 0,
            "mean_absolute_disagreement": None,
            "market_brier": None,
            "jev_brier": None,
            "delta_brier": None,
        }
    market_subset = [market_probabilities[index] for index in selected]
    jev_subset = [jev_probabilities[index] for index in selected]
    outcome_subset = [outcomes[index] for index in selected]
    market_brier = brier_score(market_subset, outcome_subset)
    jev_brier = brier_score(jev_subset, outcome_subset)
    return {
        "n": len(selected),
        "mean_absolute_disagreement": sum(
            abs(jev_probabilities[index] - market_probabilities[index])
            for index in selected
        )
        / len(selected),
        "market_brier": market_brier,
        "jev_brier": jev_brier,
        "delta_brier": market_brier - jev_brier,
    }


def ensure_no_duplicate_complete_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    counts = Counter(
        (row["asset"], row["market_slug"], row["checkpoint"]) for row in rows
    )
    duplicates = [(key, count) for key, count in counts.items() if count > 1]
    if not duplicates:
        return
    details = "; ".join(
        f"asset={key[0]}, market_slug={key[1]}, checkpoint={key[2]} (count={count})"
        for key, count in duplicates
    )
    raise AnalysisError(f"duplicate complete observations: {details}")


def _validate_metric_inputs(
    probabilities: Sequence[float], outcomes: Sequence[int]
) -> None:
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must be non-empty and equal length")


def _database_path() -> Path:
    config_path = Path(__file__).with_name("config.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not config.get("database_path"):
        raise AnalysisError("config.yaml is missing database_path")
    path = Path(str(config["database_path"]))
    return path if path.is_absolute() else Path(__file__).parent / path


def _load_dataset(database_path: Path) -> tuple[dict[str, int], list[dict[str, Any]]]:
    if not database_path.exists():
        raise AnalysisError(f"database does not exist: {database_path}")
    with duckdb.connect(str(database_path), read_only=True) as connection:
        total, excluded_unresolved, excluded_missing_jev = connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT(*) FILTER (
                    WHERE outcome IS NULL OR outcome NOT IN ('UP', 'DOWN')
                ),
                COUNT(*) FILTER (
                    WHERE outcome IN ('UP', 'DOWN')
                      AND (p_jev_blind IS NULL OR p_jev_meta IS NULL)
                )
            FROM snapshots
            WHERE checkpoint IS NOT NULL
            """
        ).fetchone()
        cursor = connection.execute(
            """
            SELECT asset, market_slug, checkpoint, outcome,
                   p_market, p_simple, p_jev_blind, p_jev_meta
            FROM snapshots
            WHERE checkpoint IS NOT NULL
              AND outcome IN ('UP', 'DOWN')
              AND p_market IS NOT NULL
              AND p_simple IS NOT NULL
              AND p_jev_blind IS NOT NULL
              AND p_jev_meta IS NOT NULL
            ORDER BY market_slug, checkpoint, asset
            """
        )
        columns = [item[0] for item in cursor.description]
        rows = [dict(zip(columns, values)) for values in cursor.fetchall()]
    counts = {
        "total_checkpoint_rows": int(total),
        "excluded_unresolved": int(excluded_unresolved),
        "excluded_missing_jev": int(excluded_missing_jev),
    }
    return counts, rows


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_no_duplicate_complete_rows(rows)
    for row in rows:
        for _label, field in MODELS:
            value = float(row[field])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise AnalysisError(
                    f"invalid probability {field}={row[field]!r} for "
                    f"{row['asset']} {row['market_slug']} {row['checkpoint']}"
                )


def _series(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows]


def _outcomes(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return [outcome_to_binary(str(row["outcome"])) for row in rows]


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def print_analysis(counts: Mapping[str, int], rows: list[dict[str, Any]]) -> None:
    unique_markets = len({(row["asset"], row["market_slug"]) for row in rows})
    print("PolyJev Stage 4 Analysis")
    print("\nDataset")
    print("-------")
    print(f"Total checkpoint rows: {counts['total_checkpoint_rows']}")
    print(f"Complete resolved rows: {len(rows)}")
    print(f"Unique complete markets: {unique_markets}")
    print(f"Excluded unresolved: {counts['excluded_unresolved']}")
    print(f"Excluded missing Jev: {counts['excluded_missing_jev']}")
    if unique_markets < 6:
        print("\nWARNING: Dataset is very small; metrics are diagnostic only.")
    if not rows:
        print("\nNo complete resolved checkpoint observations.")
        return

    outcomes = _outcomes(rows)
    market_brier = brier_score(_series(rows, "p_market"), outcomes)
    print("\nOverall")
    print("-------")
    print("Model       Brier     Delta vs Market  LogLoss")
    print("----------- ---------- ---------------- ----------")
    for label, field in MODELS:
        probabilities = _series(rows, field)
        score = brier_score(probabilities, outcomes)
        print(
            f"{label:<11} {score:>10.6f} {market_brier - score:>16.6f} "
            f"{binary_log_loss(probabilities, outcomes):>10.6f}"
        )

    print("\nBy checkpoint")
    print("-------------")
    print("Checkpoint  N  Market     Simple     Blind      Meta")
    print("---------- --- ---------- ---------- ---------- ----------")
    for checkpoint in CHECKPOINTS:
        subset = [row for row in rows if row["checkpoint"] == checkpoint]
        if subset:
            subset_outcomes = _outcomes(subset)
            values = [
                brier_score(_series(subset, field), subset_outcomes)
                for _label, field in MODELS
            ]
        else:
            values = [None] * len(MODELS)
        print(
            f"{checkpoint:<10} {len(subset):>3} "
            + " ".join(f"{_format_metric(value):>10}" for value in values)
        )

    print("\nJev disagreement")
    print("----------------")
    print("Model  |D|   N  Mean |D|  Market Brier  Jev Brier  Delta")
    print("------ ---- --- --------- ------------- ---------- ----------")
    market = _series(rows, "p_market")
    for label, field in (("Blind", "p_jev_blind"), ("Meta", "p_jev_meta")):
        jev = _series(rows, field)
        for threshold, threshold_label in ((0.05, "5pp"), (0.10, "10pp")):
            metrics = disagreement_metrics(market, jev, outcomes, threshold)
            print(
                f"{label:<6} {threshold_label:<4} {metrics['n']:>3} "
                f"{_format_metric(metrics['mean_absolute_disagreement']):>9} "
                f"{_format_metric(metrics['market_brier']):>13} "
                f"{_format_metric(metrics['jev_brier']):>10} "
                f"{_format_metric(metrics['delta_brier']):>10}"
            )


def main() -> int:
    try:
        counts, rows = _load_dataset(_database_path())
        _validate_rows(rows)
        print_analysis(counts, rows)
        return 0
    except (AnalysisError, duckdb.Error, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
